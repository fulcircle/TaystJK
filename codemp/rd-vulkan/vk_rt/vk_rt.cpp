#include "qcommon/q_math.h"
#include "qcommon/q_shared.h"
#include "tr_local.h"
#include "vk_local.h"
#include "vulkan/vulkan_core.h"
#include "arena.h"
#include "vk_rt.h"
#include <cstddef>
#include <cstdint>

#define MAX_LIGHT_INFOS 1000

typedef struct {
	uint32_t 					lightDebugMode;
	float 						falloffScale;
	int32_t						debugLightIndex;
	uint32_t					numLights;
	uint32_t					rtEnable;		// 0 = bypass RT direct lighting (lightmap/fullbright)
	uint32_t					_pad[3];		// std140: pad block to 32 bytes
} rtParams_t;

typedef struct {
	VkBuffer	   positionBuffer;
	VkDeviceMemory positionMemory;
	VkBuffer       indexBuffer;
	VkDeviceMemory indexMemory;
	uint32_t       numVertices;
	uint32_t       numIndices;

	// BLAS
	VkBuffer                   asBuffer;
	VkDeviceMemory 			   asMemory;
	VkAccelerationStructureKHR blas;
	VkDeviceAddress 		   blasAddress;

	// TLAS
	VkAccelerationStructureKHR tlas;
	VkBuffer				   tlasBuffer;
	VkDeviceMemory             tlasMemory;

	// empty TLAS (a mask=0 instance) bound for non-world draws so rays always miss
	VkAccelerationStructureKHR tlasEmpty;
	VkBuffer				   tlasEmptyBuffer;
	VkDeviceMemory             tlasEmptyMemory;

	// Lighting
	VkBuffer 				   lightBuffer;
	VkDeviceMemory             lightMemory;
	uint32_t				   numLights;

	// Lighting Params
	VkBuffer				   paramsBuffer;
	VkDeviceMemory			   paramsMemory;
	rtParams_t				   *rtParams;

} world_rt_t;

static world_rt_t world_rt;

static void vk_rt_write_params_descriptor( VkDescriptorSet set) {
	VkDescriptorBufferInfo bufInfo;
	VkWriteDescriptorSet   write;

	bufInfo.buffer = world_rt.paramsBuffer;
	bufInfo.offset = 0;
	bufInfo.range  = VK_WHOLE_SIZE;

	write.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.pNext            = NULL;
	write.dstSet           = set;
	write.dstBinding       = 2;
	write.dstArrayElement  = 0;
	write.descriptorCount  = 1;
	write.descriptorType   = VK_DESCRIPTOR_TYPE_UNIFORM_BUFFER;
	write.pImageInfo       = NULL;
	write.pBufferInfo      = &bufInfo;
	write.pTexelBufferView = NULL;

	qvkUpdateDescriptorSets( vk.device, 1, &write, 0, NULL );
}

static void vk_rt_write_light_descriptor( VkDescriptorSet set ) {
	VkDescriptorBufferInfo bufInfo;
	VkWriteDescriptorSet   write;

	bufInfo.buffer = world_rt.lightBuffer;
	bufInfo.offset = 0;
	bufInfo.range  = VK_WHOLE_SIZE;

	write.sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
	write.pNext            = NULL;
	write.dstSet           = set;
	write.dstBinding       = 1;
	write.dstArrayElement  = 0;
	write.descriptorCount  = 1;
	write.descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
	write.pImageInfo       = NULL;
	write.pBufferInfo      = &bufInfo;
	write.pTexelBufferView = NULL;

	qvkUpdateDescriptorSets( vk.device, 1, &write, 0, NULL );
}

// Create a buffer + back it with memory of the requested properties.
// memProps selects device-local vs host-visible; deviceAddress adds the
// device-address allocation flag (needed for acceleration-structure buffers).
static void vk_rt_create_buffer( VkDeviceSize size, VkBufferUsageFlags usage,
	VkMemoryPropertyFlags memProps, qboolean deviceAddress,
	VkBuffer *outBuffer, VkDeviceMemory *outMemory )
{
	VkBufferCreateInfo desc;
	VkMemoryAllocateInfo alloc_info;
	VkMemoryAllocateFlagsInfo flagsInfo;
	VkMemoryRequirements mem_reqs;

	desc.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
	desc.pNext = NULL; desc.flags = 0;
	desc.sharingMode = VK_SHARING_MODE_EXCLUSIVE;
	desc.queueFamilyIndexCount = 0; desc.pQueueFamilyIndices = NULL;
	desc.size = size;
	desc.usage = usage;
	VK_CHECK( qvkCreateBuffer( vk.device, &desc, NULL, outBuffer ) );

	qvkGetBufferMemoryRequirements( vk.device, *outBuffer, &mem_reqs );
	alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
	alloc_info.pNext = NULL;
	alloc_info.allocationSize = mem_reqs.size;
	alloc_info.memoryTypeIndex = vk_find_memory_type(mem_reqs.memoryTypeBits, memProps);
	if ( deviceAddress ) {
		flagsInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_FLAGS_INFO;
		flagsInfo.pNext = NULL;
		flagsInfo.flags = VK_MEMORY_ALLOCATE_DEVICE_ADDRESS_BIT;
		flagsInfo.deviceMask = 0;
		alloc_info.pNext = &flagsInfo;
	}
	VK_CHECK( qvkAllocateMemory ( vk.device, &alloc_info, NULL, outMemory ) );
	qvkBindBufferMemory( vk.device, *outBuffer, *outMemory, 0 );
}

static void vk_rt_upload_buffer( VkDeviceSize size, const void *src, VkBufferUsageFlags usage,
	VkBuffer *outBuffer, VkDeviceMemory *outMemory)
{
	VkBuffer                  staging;
	VkDeviceMemory            stagingMem;
	VkCommandBuffer           cmd;
	VkBufferCopy              region;
	void                      *data;

	// device-local destination (device address flag preserves prior behaviour)
	vk_rt_create_buffer( size, VK_BUFFER_USAGE_TRANSFER_DST_BIT | usage,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, qtrue, outBuffer, outMemory );

	// host-visible staging source
	vk_rt_create_buffer( size, VK_BUFFER_USAGE_TRANSFER_SRC_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		qfalse, &staging, &stagingMem );

	VK_CHECK( qvkMapMemory ( vk.device, stagingMem, 0, VK_WHOLE_SIZE, 0, &data ) );
	memcpy( data, src, (size_t)size );
	qvkUnmapMemory( vk.device, stagingMem );

	cmd = vk_begin_command_buffer();
	region.srcOffset = 0; region.dstOffset = 0; region.size = size;
	qvkCmdCopyBuffer( cmd, staging, *outBuffer, 1, &region );
	vk_end_command_buffer( cmd, __func__ );
	qvkDestroyBuffer( vk.device, staging, NULL );
	qvkFreeMemory( vk.device, stagingMem, NULL );
}

// Host-visible, persistently-mapped UBO for per-frame RT params (debug mode etc).
// Created once per world load; mapped pointer kept in world_rt.rtParams,
// destroyed in vk_rt_release_world.
static void vk_rt_create_params_buffer( void )
{
	vk_rt_create_buffer( sizeof( rtParams_t ), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		qfalse, &world_rt.paramsBuffer, &world_rt.paramsMemory );

	VK_CHECK( qvkMapMemory( vk.device, world_rt.paramsMemory, 0, VK_WHOLE_SIZE, 0, (void **)&world_rt.rtParams ) );
}

static void vk_rt_build_world_blas ( void ) {
	VkBufferDeviceAddressInfo                   addrInfo;
	VkDeviceAddress                             posAddr, idxAddr;
	VkAccelerationStructureGeometryKHR          geom;
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo;
	VkAccelerationStructureBuildSizesInfoKHR    sizeInfo;
	uint32_t                                    primCount;

	if ( !vk.rayQuery || world_rt.numIndices == 0)
		return;

	addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	addrInfo.pNext = NULL;
	addrInfo.buffer = world_rt.positionBuffer;
	posAddr = qvkGetBufferDeviceAddress( vk.device, &addrInfo );
	addrInfo.buffer = world_rt.indexBuffer;
	idxAddr = qvkGetBufferDeviceAddress( vk.device, &addrInfo );

	Com_Memset( &geom, 0, sizeof(geom) );
	geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geom.geometryType = VK_GEOMETRY_TYPE_TRIANGLES_KHR;
	geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geom.geometry.triangles.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_TRIANGLES_DATA_KHR;
	geom.geometry.triangles.vertexFormat =  VK_FORMAT_R32G32B32_SFLOAT;
	geom.geometry.triangles.vertexData.deviceAddress = posAddr;
	geom.geometry.triangles.vertexStride = sizeof(vec3_t);
	geom.geometry.triangles.maxVertex = world_rt.numVertices - 1;
	geom.geometry.triangles.indexType = VK_INDEX_TYPE_UINT32;
	geom.geometry.triangles.indexData.deviceAddress = idxAddr;

	Com_Memset(&buildInfo, 0, sizeof(buildInfo));
	buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
	buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries = &geom;

	primCount = world_rt.numIndices / 3;
	Com_Memset( &sizeInfo, 0, sizeof(sizeInfo) );
	sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	qvkGetAccelerationStructureBuildSizesKHR( vk.device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR,
		&buildInfo, &primCount, &sizeInfo);

	ri.Printf( PRINT_ALL, "...BLAS sizes: structure=%u bytes, scratch=%u bytes (%u tris)\n",
		(unsigned)sizeInfo.accelerationStructureSize,
		(unsigned)sizeInfo.buildScratchSize,
		primCount );

	vk_rt_create_buffer( sizeInfo.accelerationStructureSize,
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, qtrue, &world_rt.asBuffer, &world_rt.asMemory);
	{
		VkAccelerationStructureCreateInfoKHR asCreate;
		Com_Memset( &asCreate, 0, sizeof(asCreate) );
		asCreate.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		asCreate.buffer = world_rt.asBuffer;
		asCreate.size = sizeInfo.accelerationStructureSize;
		asCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_BOTTOM_LEVEL_KHR;
		VK_CHECK( qvkCreateAccelerationStructureKHR(vk.device, &asCreate, NULL, &world_rt.blas ) );
	}

	{
		VkBuffer       scratchBuffer;
		VkDeviceMemory scratchMemory;
		VkBufferDeviceAddressInfo                sAddr;
		VkAccelerationStructureBuildRangeInfoKHR range;
		const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;
		VkAccelerationStructureDeviceAddressInfoKHR blasAddrInfo;
		VkCommandBuffer cmd;

		vk_rt_create_buffer( sizeInfo.buildScratchSize,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, qtrue, &scratchBuffer, &scratchMemory );

		sAddr.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		sAddr.pNext = NULL;
		sAddr.buffer = scratchBuffer;

		buildInfo.dstAccelerationStructure = world_rt.blas;
		buildInfo.scratchData.deviceAddress = qvkGetBufferDeviceAddress( vk.device, &sAddr );

		range.primitiveCount = primCount;
		range.primitiveOffset = 0;
		range.firstVertex = 0;
		range.transformOffset = 0;

		cmd = vk_begin_command_buffer();
		qvkCmdBuildAccelerationStructuresKHR( cmd, 1, &buildInfo, &pRange );
		vk_end_command_buffer( cmd, __func__ );

		qvkDestroyBuffer( vk.device, scratchBuffer, NULL );
		qvkFreeMemory( vk.device, scratchMemory, NULL );

		Com_Memset( &blasAddrInfo, 0, sizeof(blasAddrInfo) );
		blasAddrInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_DEVICE_ADDRESS_INFO_KHR;
		blasAddrInfo.accelerationStructure = world_rt.blas;
		world_rt.blasAddress = qvkGetAccelerationStructureDeviceAddressKHR( vk.device, &blasAddrInfo );

	}

	ri.Printf( PRINT_ALL, "...BLAS built: address=0x%llx\n", (unsigned long long)world_rt.blasAddress );
}

static void vk_rt_build_world_tlas ( void )
{
	VkAccelerationStructureInstanceKHR          instance;
	VkBuffer                                    instBuffer;
	VkDeviceMemory                              instMemory;
	VkBufferDeviceAddressInfo                   addrInfo;
	VkAccelerationStructureGeometryKHR          geom;
	VkAccelerationStructureBuildGeometryInfoKHR buildInfo;
	VkAccelerationStructureBuildSizesInfoKHR    sizeInfo;
	uint32_t                                    instCount = 1;

	if ( !vk.rayQuery || world_rt.blas == VK_NULL_HANDLE ) {
		return;
	}

	Com_Memset( &instance, 0, sizeof(instance) );

	// identity 3x4 (row-major)
	instance.transform.matrix[0][0] = 1.0f;
	instance.transform.matrix[1][1] = 1.0f;
	instance.transform.matrix[2][2] = 1.0f;
	instance.mask = 0xFF;
	instance.flags = VK_GEOMETRY_INSTANCE_TRIANGLE_FACING_CULL_DISABLE_BIT_KHR;
	instance.accelerationStructureReference = world_rt.blasAddress;

	vk_rt_upload_buffer( sizeof(instance), &instance,
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		&instBuffer, &instMemory );
	addrInfo.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
	addrInfo.pNext = NULL;
	addrInfo.buffer = instBuffer;

	Com_Memset( &geom, 0, sizeof(geom) );
	geom.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_KHR;
	geom.geometryType = VK_GEOMETRY_TYPE_INSTANCES_KHR;
	geom.flags = VK_GEOMETRY_OPAQUE_BIT_KHR;
	geom.geometry.instances.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_GEOMETRY_INSTANCES_DATA_KHR;
	geom.geometry.instances.arrayOfPointers = VK_FALSE;
	geom.geometry.instances.data.deviceAddress = qvkGetBufferDeviceAddress( vk.device, &addrInfo );

	Com_Memset( &buildInfo, 0, sizeof(buildInfo) );
	buildInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_GEOMETRY_INFO_KHR;
	buildInfo.mode = VK_BUILD_ACCELERATION_STRUCTURE_MODE_BUILD_KHR;
	buildInfo.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
	buildInfo.flags = VK_BUILD_ACCELERATION_STRUCTURE_PREFER_FAST_TRACE_BIT_KHR;
	buildInfo.geometryCount = 1;
	buildInfo.pGeometries = &geom;

	Com_Memset( &sizeInfo, 0, sizeof(sizeInfo) );
	sizeInfo.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
	qvkGetAccelerationStructureBuildSizesKHR( vk.device,
		VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &instCount, &sizeInfo );

	vk_rt_create_buffer( sizeInfo.accelerationStructureSize,
		VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
		VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, qfalse, &world_rt.tlasBuffer, &world_rt.tlasMemory );
	{
		VkAccelerationStructureCreateInfoKHR asCreate;
		Com_Memset( &asCreate, 0, sizeof(asCreate) );
		asCreate.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
		asCreate.buffer = world_rt.tlasBuffer;
		asCreate.size = sizeInfo.accelerationStructureSize;
		asCreate.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
		VK_CHECK( qvkCreateAccelerationStructureKHR (vk.device, &asCreate, NULL, &world_rt.tlas ) );
	}

	{
		VkBuffer scratchBuffer; VkDeviceMemory scratchMemory;
		VkBufferDeviceAddressInfo sAddr;
		VkAccelerationStructureBuildRangeInfoKHR range;
		const VkAccelerationStructureBuildRangeInfoKHR *pRange = &range;
		VkCommandBuffer cmd;

		vk_rt_create_buffer( sizeInfo.buildScratchSize,
			VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, qtrue, &scratchBuffer, &scratchMemory);
		sAddr.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		sAddr.pNext = NULL;
		sAddr.buffer = scratchBuffer;

		buildInfo.dstAccelerationStructure = world_rt.tlas;
		buildInfo.scratchData.deviceAddress = qvkGetBufferDeviceAddress( vk.device, &sAddr );

		range.primitiveCount = 1;
		range.primitiveOffset = 0; range.firstVertex = 0; range.transformOffset = 0;

		cmd = vk_begin_command_buffer();
		qvkCmdBuildAccelerationStructuresKHR( cmd, 1, &buildInfo, &pRange);
		vk_end_command_buffer( cmd, __func__ );

		qvkDestroyBuffer( vk.device, scratchBuffer, NULL );
		qvkFreeMemory( vk.device, scratchMemory, NULL );
	}

	qvkDestroyBuffer( vk.device, instBuffer, NULL );
	qvkFreeMemory( vk.device, instMemory, NULL );

	{
		VkWriteDescriptorSetAccelerationStructureKHR asInfo;
		VkWriteDescriptorSet write;

		asInfo.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
		asInfo.pNext = NULL;
		asInfo.accelerationStructureCount = 1;
		asInfo.pAccelerationStructures = &world_rt.tlas;

		write.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write.pNext = &asInfo;
		write.dstSet = vk.descriptor_rt;
		write.dstBinding = 0;
		write.dstArrayElement = 0;
		write.descriptorCount = 1;
		write.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
		write.pImageInfo = NULL;
		write.pBufferInfo = NULL;
		write.pTexelBufferView = NULL;

		qvkUpdateDescriptorSets( vk.device, 1, &write, 0, NULL );
	}

	// --- empty TLAS: same build, but the single instance's mask is 0, so any ray
	// (cullMask 0xFF) is culled -> always a miss. Bound for non-world draws (models/2D)
	// whose var_WorldPos isn't world space, so they get no RT shadow.
	{
		VkAccelerationStructureInstanceKHR       einst = instance;
		VkBuffer                                 einstBuffer;
		VkDeviceMemory                           einstMemory;
		VkBufferDeviceAddressInfo                eAddr;
		VkAccelerationStructureBuildSizesInfoKHR eSize;
		uint32_t                                 eCount = 1;

		einst.mask = 0;
		vk_rt_upload_buffer( sizeof(einst), &einst,
			VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
			&einstBuffer, &einstMemory );
		eAddr.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
		eAddr.pNext = NULL;
		eAddr.buffer = einstBuffer;
		geom.geometry.instances.data.deviceAddress = qvkGetBufferDeviceAddress( vk.device, &eAddr );

		Com_Memset( &eSize, 0, sizeof(eSize) );
		eSize.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_BUILD_SIZES_INFO_KHR;
		qvkGetAccelerationStructureBuildSizesKHR( vk.device,
			VK_ACCELERATION_STRUCTURE_BUILD_TYPE_DEVICE_KHR, &buildInfo, &eCount, &eSize );

		vk_rt_create_buffer( eSize.accelerationStructureSize,
			VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_STORAGE_BIT_KHR,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, qfalse, &world_rt.tlasEmptyBuffer, &world_rt.tlasEmptyMemory );
		{
			VkAccelerationStructureCreateInfoKHR ec;
			Com_Memset( &ec, 0, sizeof(ec) );
			ec.sType = VK_STRUCTURE_TYPE_ACCELERATION_STRUCTURE_CREATE_INFO_KHR;
			ec.buffer = world_rt.tlasEmptyBuffer;
			ec.size = eSize.accelerationStructureSize;
			ec.type = VK_ACCELERATION_STRUCTURE_TYPE_TOP_LEVEL_KHR;
			VK_CHECK( qvkCreateAccelerationStructureKHR( vk.device, &ec, NULL, &world_rt.tlasEmpty ) );
		}
		{
			VkBuffer es; VkDeviceMemory esm;
			VkBufferDeviceAddressInfo esa;
			VkAccelerationStructureBuildRangeInfoKHR er;
			const VkAccelerationStructureBuildRangeInfoKHR *epr = &er;
			VkCommandBuffer ecmd;

			vk_rt_create_buffer( eSize.buildScratchSize,
				VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT,
				VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, qtrue, &es, &esm );
			esa.sType = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO;
			esa.pNext = NULL;
			esa.buffer = es;

			buildInfo.dstAccelerationStructure = world_rt.tlasEmpty;
			buildInfo.scratchData.deviceAddress = qvkGetBufferDeviceAddress( vk.device, &esa );

			er.primitiveCount = 1;
			er.primitiveOffset = 0; er.firstVertex = 0; er.transformOffset = 0;

			ecmd = vk_begin_command_buffer();
			qvkCmdBuildAccelerationStructuresKHR( ecmd, 1, &buildInfo, &epr );
			vk_end_command_buffer( ecmd, __func__ );

			qvkDestroyBuffer( vk.device, es, NULL );
			qvkFreeMemory( vk.device, esm, NULL );
		}
		qvkDestroyBuffer( vk.device, einstBuffer, NULL );
		qvkFreeMemory( vk.device, einstMemory, NULL );

		{
			VkWriteDescriptorSetAccelerationStructureKHR ai;
			VkWriteDescriptorSet w;

			ai.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET_ACCELERATION_STRUCTURE_KHR;
			ai.pNext = NULL;
			ai.accelerationStructureCount = 1;
			ai.pAccelerationStructures = &world_rt.tlasEmpty;

			w.sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
			w.pNext = &ai;
			w.dstSet = vk.descriptor_rt_empty;
			w.dstBinding = 0;
			w.dstArrayElement = 0;
			w.descriptorCount = 1;
			w.descriptorType = VK_DESCRIPTOR_TYPE_ACCELERATION_STRUCTURE_KHR;
			w.pImageInfo = NULL;
			w.pBufferInfo = NULL;
			w.pTexelBufferView = NULL;

			qvkUpdateDescriptorSets( vk.device, 1, &w, 0, NULL );
		}
	}

	ri.Printf( PRINT_ALL, "...TLAS built (1 instance)\n" );
}


void vk_rt_release_world( void )
{

	if ( world_rt.tlas ) {
		qvkDestroyAccelerationStructureKHR ( vk.device, world_rt.tlas, NULL );
	}
	if (world_rt.tlasBuffer) {
		qvkDestroyBuffer( vk.device, world_rt.tlasBuffer, NULL );
		qvkFreeMemory( vk.device, world_rt.tlasMemory, NULL );
	}

	if ( world_rt.tlasEmpty ) {
		qvkDestroyAccelerationStructureKHR ( vk.device, world_rt.tlasEmpty, NULL );
	}
	if ( world_rt.tlasEmptyBuffer ) {
		qvkDestroyBuffer( vk.device, world_rt.tlasEmptyBuffer, NULL );
		qvkFreeMemory( vk.device, world_rt.tlasEmptyMemory, NULL );
	}

	if ( world_rt.blas ) {
		qvkDestroyAccelerationStructureKHR( vk.device, world_rt.blas, NULL );
	}
	if ( world_rt.asBuffer ) {
		qvkDestroyBuffer( vk.device, world_rt.asBuffer, NULL );
		qvkFreeMemory ( vk.device, world_rt.asMemory, NULL );
	}

	if ( world_rt.positionBuffer ) {
		qvkDestroyBuffer( vk.device, world_rt.positionBuffer, NULL);
		qvkFreeMemory( vk.device, world_rt.positionMemory, NULL );
	}
	if ( world_rt.indexBuffer ) {
		qvkDestroyBuffer( vk.device, world_rt.indexBuffer, NULL );
		qvkFreeMemory( vk.device, world_rt.indexMemory, NULL );
	}

	if ( world_rt.lightBuffer ) {
		qvkDestroyBuffer( vk.device, world_rt.lightBuffer, NULL );
		qvkFreeMemory( vk.device, world_rt.lightMemory, NULL );
	}

	if ( world_rt.paramsBuffer ) {
		qvkDestroyBuffer( vk.device, world_rt.paramsBuffer, NULL );
		qvkFreeMemory( vk.device, world_rt.paramsMemory, NULL );
	}

	Com_Memset( &world_rt, 0, sizeof(world_rt) );
}

#define MAX_RT_LIGHTS 1024


static void R_rtSynthesizeSurfaceLights( world_t &worldData ) {
	backEnd.currentEntity = &tr.worldEntity;

	msurface_t *surfaces = worldData.surfaces;
	int numsurfaces = worldData.numsurfaces;
	vec3_t white = {1, 1, 1};
	
	arena_t lightArena = arena_init(sizeof(rtLight_t), alignof(rtLight_t));
	for ( int i = 0; i < numsurfaces; i++ ) {
		
		msurface_t *surface = &surfaces[i];
		if (surface->shader == NULL || surface->shader->surfaceLight <= 0.0f) {
			continue;
		}

		int type = *surface->data;
		if ( type != SF_FACE && type != SF_TRIANGLES && type != SF_GRID ) {
			continue;
		}

		if ( worldData.numStaticLights + lightArena.numElements >= MAX_RT_LIGHTS ) {  // no silent cap
			ri.Printf( PRINT_WARNING, "RT: MAX_RT_LIGHTS hit, skipping rest of surface lights\n" );
			break;
		}

		RB_BeginSurface(surface->shader, 0);
		tess.allowVBO = qfalse;
		rb_surfaceTable[type]( surface->data );

		if ( tess.numVertexes == 0 ) {
			tess.numIndexes = 0;
			continue;
		}
		
		for ( int k = 0; k + 2 < tess.numIndexes; k += 3 ) {
			rtLight_t *polygon = (rtLight_t *)arena_alloc(&lightArena);
			
			polygon->type = LIGHT_TYPE_POLYGON;
			VectorScale(white, surface->shader->surfaceLight, polygon->color);
			
			VectorCopy(tess.xyz[ tess.indexes[k + 0] ], polygon->positions + 0 );
			VectorCopy(tess.xyz[ tess.indexes[k + 1] ], polygon->positions + 3);
			VectorCopy(tess.xyz[ tess.indexes[k + 2] ], polygon->positions + 6);
		} 

		tess.numVertexes = 0;
		tess.numIndexes = 0;
	}
	
	worldData.numStaticLights = lightArena.numElements;
	worldData.rtStaticLights = NULL;
	if (lightArena.numElements > 0) {
		worldData.rtStaticLights =
			(rtLight_t *)Hunk_Alloc( lightArena.used, h_low );
	
		memcpy( worldData.rtStaticLights, lightArena.base, lightArena.used);

	}
	ri.Printf( PRINT_ALL, "RT: synthesized %u surface lights from %u surfaces\n", lightArena.numElements, numsurfaces );
	
	arena_free(&lightArena);
}

static void R_rtBuildWorldLightBuffers( rtLight_t *staticLights, uint32_t numLights ) {
	rtLight_t  dummyLight = {};
	rtLight_t *lights;
	if (!vk.rayQuery) return;
	if (numLights == 0) {
		// Bind one zeroed dummy light so binding 1 is always a valid descriptor.
		// The shader loops on u_numLights (0 here), so the dummy is never read.
		world_rt.numLights = 0;
		numLights = 1;
		lights = &dummyLight;
	} else {
		world_rt.numLights = numLights;
		lights = staticLights;
	}

	if (world_rt.rtParams) world_rt.rtParams->numLights = world_rt.numLights;

	VkDeviceSize size = numLights * sizeof(rtLight_t);
	vk_rt_upload_buffer(size, lights, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT, &world_rt.lightBuffer, &world_rt.lightMemory);

	ri.Printf( PRINT_ALL, "..RT lights uploaded: %u\n", numLights);

	vk_rt_write_light_descriptor( vk.descriptor_rt );
	vk_rt_write_light_descriptor(vk.descriptor_rt_empty);

}

void R_rtBuildWorldLights( world_t &worldData ) {
	R_rtSynthesizeSurfaceLights( worldData );
	R_rtBuildWorldLightBuffers( worldData.rtStaticLights, worldData.numStaticLights );
}

void R_rtBuildWorldGeometryBuffers(msurface_t *surf, int surfCount) {
	msurface_t *sf;
	int i, k, type;
	int numVertexes = 0, numIndexes = 0;
	vec3_t *positions;
	uint32_t *indices;
	uint32_t baseVertex = 0, baseIndex = 0;
	vec3_t mins, maxs;

	if (!vk.rayQuery) {
		return;
	}

	vk_rt_release_world();

	for (i = 0, sf = surf; i < surfCount; i++, sf++) {
		switch( *sf->data ) {
			case SF_FACE: {
				srfSurfaceFace_t *face = (srfSurfaceFace_t *)sf->data;
				numVertexes += face->numPoints; numIndexes += face->numIndices;
				break;
			}
			case SF_TRIANGLES: {
				srfTriangles_t *tris = (srfTriangles_t *)sf->data;
				numVertexes += tris->numVerts; numIndexes += tris->numIndexes;
				break;
			}
			case SF_GRID: {
				srfGridMesh_t *grid = (srfGridMesh_t *)sf->data;
				int gv, gi;
				RB_SurfaceGridEstimate(grid, &gv, &gi);
				numVertexes += gv; numIndexes += gi;
				break;
			}
			default:
				break;
		}
	}
	if (numVertexes == 0) {
		ri.Printf(PRINT_ALL, "...no RT geometry\n");
		return;
	}

	positions = (vec3_t *)ri.Hunk_AllocateTempMemory(numVertexes * sizeof(vec3_t));
	indices = (uint32_t *)ri.Hunk_AllocateTempMemory(numIndexes * sizeof(uint32_t));

	Com_Memset(&backEnd.viewParms, 0, sizeof(backEnd.viewParms));
	backEnd.currentEntity = &tr.worldEntity;

	for ( i = 0, sf = surf; i < surfCount; i++, sf++) {
		type = *sf->data;
		if (type != SF_FACE && type != SF_TRIANGLES && type != SF_GRID)
			continue;

		RB_BeginSurface(sf->shader, 0);
		tess.allowVBO = qfalse;
		rb_surfaceTable[type]( sf-> data);

		for ( k = 0; k < tess.numVertexes; k++) {
			VectorCopy(tess.xyz[k], positions[baseVertex + k]);
		}
		for ( k = 0; k < tess.numIndexes; k++) {
			indices[baseIndex + k] = baseVertex + tess.indexes[k];
		}

		baseVertex += tess.numVertexes;
		baseIndex += tess.numIndexes;
		tess.numVertexes = 0;
		tess.numIndexes = 0;
	}

	// sanity: bbox of all collected positions
	ClearBounds(mins, maxs);
	for ( i = 0; i < (int)baseVertex; i++ ) {
		AddPointToBounds( positions[i], mins, maxs );
	}

	ri.Printf( PRINT_ALL, "...RT geom collected: %u verts, %u tris; bbox (%.0f %.0f %.0f)...(%.0f %.0f %.0f)\n",
			baseVertex, baseIndex / 3,
			mins[0], mins[1], mins[2], maxs[0], maxs[1], maxs[2] );

	vk_rt_upload_buffer((VkDeviceSize)baseVertex * sizeof(vec3_t), positions,
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		&world_rt.positionBuffer, &world_rt.positionMemory);
	vk_rt_upload_buffer( (VkDeviceSize)baseIndex * sizeof(uint32_t), indices,
		VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT | VK_BUFFER_USAGE_ACCELERATION_STRUCTURE_BUILD_INPUT_READ_ONLY_BIT_KHR,
		&world_rt.indexBuffer, &world_rt.indexMemory );
	world_rt.numVertices = baseVertex;
	world_rt.numIndices = baseIndex;

	ri.Printf( PRINT_ALL, "...RT buffers uploaded: %u verts (%u KB), %u indices (%u KB)\n",
		baseVertex, (unsigned)(baseVertex * sizeof(vec3_t) / 1024),
		baseIndex,  (unsigned)(baseIndex * sizeof(uint32_t) / 1024) );

	ri.Hunk_FreeTempMemory( indices );
	ri.Hunk_FreeTempMemory( positions );

	vk_rt_build_world_blas();
	vk_rt_build_world_tlas();
	
	vk_rt_create_params_buffer();
	vk_rt_write_params_descriptor(vk.descriptor_rt );
	vk_rt_write_params_descriptor(vk.descriptor_rt_empty );
}

void R_rtUpdateParams( void ) {
	if (!vk.rayQuery || world_rt.rtParams == NULL) {
		return;
	}

	world_rt.rtParams->lightDebugMode = r_rtDebugLighting->integer;
	world_rt.rtParams->falloffScale = 2.0f;
	world_rt.rtParams->debugLightIndex = r_rtDebugLightIndex->integer;
	world_rt.rtParams->rtEnable = r_rtEnable->integer;
	
}
