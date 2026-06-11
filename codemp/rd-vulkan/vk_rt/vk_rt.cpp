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

static float prevMvp[16];
static float prevFrameCount = 0;
static qboolean prevMvpValid = qfalse;

typedef struct {
	float						prevMvp[16];
	float 						falloffScale;
	float						surfaceLightScale;
	uint32_t					numLights;
	uint32_t					rtEnable;		// 0 = bypass RT direct lighting (lightmap/fullbright)
	uint32_t					frameCount;
	uint32_t					unused;
	uint32_t					padding;
	uint32_t					readIndex;
	uint32_t					writeIndex;
	uint32_t					width;
	uint32_t					height;
	uint32_t					padding2;
} rtParams_t;

typedef struct {
	vec3_t mins;
	vec3_t maxs;
} rtAABB_t;

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
	rtLight_t				   *mappedLights;

	// Lighting Params
	VkBuffer				   paramsBuffer;
	VkDeviceMemory			   paramsMemory;
	rtParams_t				   *rtParams;

	// ReSTIR
	VkBuffer				   reservoirBuffers[2];
	VkDeviceMemory			   reservoirMemory[2];
	uint32_t				   currentReservoirWriteIndex;

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
static void vk_rt_create_reservoir_buffers( void ) {
	VkDeviceSize size = (VkDeviceSize)glConfig.vidWidth * glConfig.vidHeight * 24;
	for ( int i = 0; i < 2; i++ ) {
		vk_rt_create_buffer( size, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT,
			VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT, qfalse,
			&world_rt.reservoirBuffers[i], &world_rt.reservoirMemory[i] );

		// Clear the buffer to 0 to prevent garbage/NaN values on level load
		VkCommandBuffer cmd = vk_begin_command_buffer();
		qvkCmdFillBuffer( cmd, world_rt.reservoirBuffers[i], 0, size, 0 );
		vk_end_command_buffer( cmd, __func__ );
	}
	world_rt.currentReservoirWriteIndex = 0;
}

static void vk_rt_write_reservoir_buffers( VkDescriptorSet set ) {
	VkDescriptorBufferInfo bufInfo[2];
	VkWriteDescriptorSet   write[2];

	for ( int i = 0; i < 2; i++ ) {
		bufInfo[i].buffer = world_rt.reservoirBuffers[i];
		bufInfo[i].offset = 0;
		bufInfo[i].range  = VK_WHOLE_SIZE;

		write[i].sType            = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET;
		write[i].pNext            = NULL;
		write[i].dstSet           = set;
		write[i].dstBinding       = 3 + i;
		write[i].dstArrayElement  = 0;
		write[i].descriptorCount  = 1;
		write[i].descriptorType   = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
		write[i].pImageInfo       = NULL;
		write[i].pBufferInfo      = &bufInfo[i];
		write[i].pTexelBufferView = NULL;
	}

	qvkUpdateDescriptorSets( vk.device, 2, write, 0, NULL );
}

static void vk_rt_create_params_buffer( void )
{
	vk_rt_create_buffer( sizeof( rtParams_t ), VK_BUFFER_USAGE_UNIFORM_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		qfalse, &world_rt.paramsBuffer, &world_rt.paramsMemory );

	VK_CHECK( qvkMapMemory( vk.device, world_rt.paramsMemory, 0, VK_WHOLE_SIZE, 0, (void **)&world_rt.rtParams ) );

	vk_rt_create_reservoir_buffers();
	vk_rt_write_reservoir_buffers( vk.descriptor_rt );
	vk_rt_write_reservoir_buffers( vk.descriptor_rt_empty );
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

	for ( int i = 0; i < 2; i++ ) {
		if ( world_rt.reservoirBuffers[i] ) {
			qvkDestroyBuffer( vk.device, world_rt.reservoirBuffers[i], NULL );
			qvkFreeMemory( vk.device, world_rt.reservoirMemory[i], NULL );
		}
	}

	prevFrameCount = 0;
	prevMvpValid = qfalse;
	Com_Memset( prevMvp, 0, sizeof(prevMvp) );

	Com_Memset( &world_rt, 0, sizeof(world_rt) );
}

#define MAX_RT_LIGHTS 1024


static void R_rtGenerateWorldLights( world_t &worldData ) {
	backEnd.currentEntity = &tr.worldEntity;

	msurface_t *surfaces = worldData.surfaces;
	int numsurfaces = worldData.numsurfaces;
	vec3_t white = {1, 1, 1};
	
	arena_t lights = { 0 };
	arena_t lightClusters = { 0 };

	for ( int i = 0; i < numsurfaces; i++ ) {
		
		msurface_t *surface = &surfaces[i];
		if (surface->shader == NULL || surface->shader->surfaceLight <= 0.0f) {
			continue;
		}

		int type = *surface->data;
		if ( type != SF_FACE && type != SF_TRIANGLES && type != SF_GRID ) {
			continue;
		}

		if ( worldData.numStaticLights + lights.numElements >= MAX_RT_LIGHTS ) {  // no silent cap
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
			rtLight_t *polygon = PushStruct(&lights, rtLight_t);
			
			polygon->type = LIGHT_TYPE_POLYGON;
			VectorScale(white, surface->shader->surfaceLight, polygon->color);

			vec3_t v0, v1, v2, normal;
			VectorCopy(tess.xyz[ tess.indexes [k + 0] ], v0);
			VectorCopy(tess.xyz[ tess.indexes [k + 1] ], v1);
			VectorCopy(tess.xyz[ tess.indexes [k + 2] ], v2);
			
			VectorCopy(v0, polygon->positions + 0);
			VectorCopy(v1, polygon->positions + 3);
			VectorCopy(v2, polygon->positions + 6);

			polygon->lightCentroid[0] = (v0[0] + v1[0] + v2[0]) / 3.0f;
			polygon->lightCentroid[1] = (v0[1] + v1[1] + v2[1]) / 3.0f;
			polygon->lightCentroid[2] = (v0[2] + v1[2] + v2[2]) / 3.0f;

			// Bounding Radius
			float dist1 = Distance(polygon->lightCentroid, v0);
			float dist2 = Distance(polygon->lightCentroid, v1);
			float dist3 = Distance(polygon->lightCentroid, v2);

			polygon->boundingRadius = MAX( MAX(dist1, dist2), dist3 );

			// Area
			vec3_t e1, e2, crossProd;
			VectorSubtract(v1, v0, e1);
			VectorSubtract(v2, v0, e2);
			CrossProduct(e1, e2, crossProd);

			polygon->area = 0.5f * VectorLength(crossProd);

			// Normalize cross product to get the triangle normal
			VectorNormalize2(crossProd, polygon->normal);

			// Lookup and store cluster the light centroid belongs to
			uint32_t leafNum = ri.CM_PointLeafnum(polygon->lightCentroid);
			uint32_t clusterId = ri.CM_LeafCluster(leafNum);
			uint32_t *lightCluster = PushStruct(&lightClusters, uint32_t);
			*lightCluster = clusterId;
		} 

		tess.numVertexes = 0;
		tess.numIndexes = 0;
	}

	rtAABB_t *clusterAABBs = (rtAABB_t *)Z_Malloc(worldData.numClusters * sizeof(rtAABB_t), TAG_TEMP_WORKSPACE, qtrue, __alignof(rtAABB_t));
	for (int i = 0; i < worldData.numClusters; i++) {
		clusterAABBs[i].mins[0] = INFINITY;
		clusterAABBs[i].mins[1] = INFINITY;
		clusterAABBs[i].mins[2] = INFINITY;

		clusterAABBs[i].maxs[0] = -INFINITY;
		clusterAABBs[i].maxs[1] = -INFINITY;
		clusterAABBs[i].maxs[2] = -INFINITY;
	}

	for (int i = worldData.numDecisionNodes; i < worldData.numnodes; i++) {
		mnode_t node = worldData.nodes[i];
		int clusterIndex = node.cluster;
		if (clusterIndex >= 0 && clusterIndex < worldData.numClusters) {
			float *currentMins = clusterAABBs[clusterIndex].mins;
			currentMins[0] = MIN(node.mins[0], currentMins[0]);
			currentMins[1] = MIN(node.mins[1], currentMins[1]);
			currentMins[2] = MIN(node.mins[2], currentMins[2]);

			float *currentMaxs = clusterAABBs[clusterIndex].maxs; 
			currentMaxs[0] = MAX(node.maxs[0], currentMaxs[0]);
			currentMaxs[1] = MAX(node.maxs[1], currentMaxs[1]);
			currentMaxs[2] = MAX(node.maxs[2], currentMaxs[2]);
		}
	}

	/*
		* for each cluster:
		*   for each light:
		*		get this light's cluster
		*       if this light's cluster is in the PVS of this outer-loop cluster
		*          add light to PVS set for this cluster
	*/

	arena_t lightListLights = { 0 };
	arena_t lightListOffsets = { 0 };

	uint32_t *lightHomeClusters = GetBuffer(&lightClusters, uint32_t);
	rtLight_t *lights_buffer = GetBuffer(&lights, rtLight_t);

	for ( int targetCluster = 0; targetCluster < worldData.numClusters; targetCluster++ ) {
		// Record the start index for targetCluster's light list
		uint32_t *offsetSlot = PushStruct(&lightListOffsets, uint32_t);
		*offsetSlot = lightListLights.numElements;

		const byte *pvs = ri.CM_ClusterPVS(targetCluster);

		for ( int lightIndex = 0; lightIndex < (int)lights.numElements; lightIndex++ ) {
			uint32_t lightHomeCluster = lightHomeClusters[lightIndex];

			// --- Check 1: PVS Visibility Check ---
			if ( pvs == NULL || ( pvs[lightHomeCluster >> 3] & ( 1 << ( lightHomeCluster & 7 ) ) ) != 0 ) {
				rtLight_t *light = &lights_buffer[lightIndex];

				 const float *mins = clusterAABBs[targetCluster].mins;
				 const float *maxs = clusterAABBs[targetCluster].maxs;

				 // --- Check 2: Plane Culling Check (Back-Facing Cull) ---
				 // p = farthest point along the direction of normal (p-vertex)
				 vec3_t p;
				 p[0] = (light->normal[0] >= 0.0f) ? maxs[0] : mins[0];
				 p[1] = (light->normal[1] >= 0.0f) ? maxs[1] : mins[1];
				 p[2] = (light->normal[2] >= 0.0f) ? maxs[2] : mins[2];

				  vec3_t dir;
				  VectorSubtract(p, light->positions, dir); // dir = p - light->positions (first vertex)
				  if (DotProduct(light->normal, dir) < 0.0f) {
					  continue; // Cull: behind the light's plane
				  }

				  // --- Check 3: Distance Culling Check (Sphere-Box Intersection) ---
				  // Find shortest squared distance from the AABB to the light's centroid
				  float sqDist = 0.0f;
				  for ( int axis = 0; axis < 3; axis++ ) {
					  float centroidVal = light->lightCentroid[axis];
					  if ( centroidVal < mins[axis] ) {
						  float d = mins[axis] - centroidVal;
						  sqDist += d * d;
					  } else if ( centroidVal > maxs[axis] ) {
						  float d = centroidVal - maxs[axis];
						  sqDist += d * d;
					  }
				  }

				  /* light sphere radius */
				  float maxIntensity = MAX(MAX(light->color[0], light->color[1]), light->color[2]);
				  float cullRadiusSq = maxIntensity * r_rtFalloffScale->value * 20.0f;

				  if (cullRadiusSq > 4000000.0f) {
					  cullRadiusSq = 4000000.0f;
				  }

				  float cullRadius = sqrtf(cullRadiusSq);
				  float totalRadius = cullRadius + light->boundingRadius;

				  if ( sqDist > totalRadius * totalRadius ) {
					  continue; // Cull: too far away!
				  }

				 uint32_t *lightListSlot = PushStruct(&lightListLights, uint32_t);
				 *lightListSlot = lightIndex;
			}
		}
	}

	// Record the final offset to define the end of the last cluster's light list
	uint32_t *finalOffsetSlot = PushStruct(&lightListOffsets, uint32_t);
	*finalOffsetSlot = lightListLights.numElements;

	worldData.numStaticLights = lights.numElements;
	worldData.rtStaticLights = NULL;
	if (lights.numElements > 0) {
		worldData.rtStaticLights =
			(rtLight_t *)Hunk_Alloc( lights.used, h_low );
	
		memcpy( worldData.rtStaticLights, GetBuffer(&lights, rtLight_t), lights.used);

	}

	ri.Printf( PRINT_ALL, "RT: synthesized %u surface lights from %u surfaces\n", lights.numElements, numsurfaces );
	
	Clear(&lights);
	Clear(&lightClusters);
	Clear(&lightListLights);
	Clear(&lightListOffsets);

	Z_Free( clusterAABBs );
}

static void R_rtBuildWorldLightBuffers( rtLight_t *staticLights, uint32_t numLights ) {
	if (!vk.rayQuery) return;

	// Create a host-visible dynamic buffer of MAX_RT_LIGHTS size
	vk_rt_create_buffer( MAX_RT_LIGHTS * sizeof(rtLight_t),
		VK_BUFFER_USAGE_STORAGE_BUFFER_BIT,
		VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
		qfalse, &world_rt.lightBuffer, &world_rt.lightMemory );

	VK_CHECK( qvkMapMemory( vk.device, world_rt.lightMemory, 0, VK_WHOLE_SIZE, 0, (void **)&world_rt.mappedLights ) );

	// Initialize the buffer with the initial static lights
	world_rt.numLights = numLights;
	if ( numLights > 0 ) {
		uint32_t uploadCount = (numLights < MAX_RT_LIGHTS ? numLights : MAX_RT_LIGHTS);
		Com_Memcpy( world_rt.mappedLights, staticLights, uploadCount * sizeof(rtLight_t) );
	} else {
		// Zero the first light to keep it valid
		Com_Memset( world_rt.mappedLights, 0, sizeof(rtLight_t) );
	}

	if (world_rt.rtParams) world_rt.rtParams->numLights = world_rt.numLights;

	ri.Printf( PRINT_ALL, "..RT lights buffer created (size: %u KB)\n", (unsigned)(MAX_RT_LIGHTS * sizeof(rtLight_t) / 1024));

	vk_rt_write_light_descriptor( vk.descriptor_rt );
	vk_rt_write_light_descriptor(vk.descriptor_rt_empty);
}

void R_rtBuildWorldLights( world_t &worldData ) {
	R_rtGenerateWorldLights( worldData );
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

	if ( backEnd.viewParms.viewportWidth != glConfig.vidWidth || backEnd.viewParms.viewportHeight != glConfig.vidHeight ) {
		return;
	}

	float mvp[16];
	float projm[16];

	Com_Memcpy(projm, backEnd.viewParms.projectionMatrix, sizeof(float)*16);
	projm[5] = -projm[5];

	myGlMultMatrix(backEnd.viewParms.world.modelViewMatrix, projm, mvp);

	if (!prevMvpValid) {
		Com_Memcpy(prevMvp, mvp, sizeof(float)*16);
		prevMvpValid = qtrue;
	}

	// CPU culling of static lights based on camera distance
	int activeCount = 0;
	if ( tr.world && tr.world->numStaticLights > 0 && world_rt.mappedLights != NULL ) {
		vec3_t camOrigin;
		VectorCopy( backEnd.viewParms.ori.origin, camOrigin );
		float cullRad = r_rtLightCullRadius->value;
		qboolean bypassCull = (cullRad <= 0.0f) ? qtrue : qfalse;

		struct CulledLight {
			rtLight_t* light;
			float dist;
		};
		static CulledLight culled[MAX_RT_LIGHTS];
		int culledCount = 0;

		for ( int i = 0; i < (int)tr.world->numStaticLights; i++ ) {
			rtLight_t *light = &tr.world->rtStaticLights[i];

			float dx = camOrigin[0] - light->lightCentroid[0];
			float dy = camOrigin[1] - light->lightCentroid[1];
			float dz = camOrigin[2] - light->lightCentroid[2];
			float dist = sqrtf( dx * dx + dy * dy + dz * dz );

			if ( bypassCull || dist <= cullRad + light->boundingRadius ) {
				culled[culledCount].light = light;
				culled[culledCount].dist = dist;
				culledCount++;
				if ( culledCount >= MAX_RT_LIGHTS ) {
					break;
				}
			}
		}

		if ( culledCount > 0 ) {
			qsort( culled, culledCount, sizeof( CulledLight ), []( const void *a, const void *b ) -> int {
				float distA = ((const CulledLight*)a)->dist;
				float distB = ((const CulledLight*)b)->dist;
				return (distA < distB) ? -1 : ((distA > distB) ? 1 : 0);
			} );

			for ( int i = 0; i < culledCount; i++ ) {
				Com_Memcpy( &world_rt.mappedLights[i], culled[i].light, sizeof(rtLight_t) );
			}
			activeCount = culledCount;
		} else {
			Com_Memset( world_rt.mappedLights, 0, sizeof(rtLight_t) );
		}
	}

	world_rt.rtParams->rtEnable = r_rtEnable->integer;
	world_rt.rtParams->falloffScale = r_rtFalloffScale->value;
	world_rt.rtParams->surfaceLightScale = r_rtSurfaceLightScale->value;
	world_rt.rtParams->frameCount = tr.frameCount;
	world_rt.rtParams->numLights = activeCount;
	world_rt.rtParams->unused = 0;
	world_rt.rtParams->readIndex = tr.frameCount % 2;
	world_rt.rtParams->writeIndex = (tr.frameCount + 1) % 2;
	world_rt.rtParams->width = glConfig.vidWidth;
	world_rt.rtParams->height = glConfig.vidHeight;
	Com_Memcpy(world_rt.rtParams->prevMvp, prevMvp, sizeof(float)*16);

	if (tr.frameCount == 0 || prevFrameCount < tr.frameCount) {
		Com_Memcpy(prevMvp, mvp, sizeof(float)*16);
		prevFrameCount = tr.frameCount;
	}
}

uint32_t R_rtGetActiveLightCount( void ) {
	if ( world_rt.rtParams ) {
		return world_rt.rtParams->numLights;
	}
	return 0;
}
