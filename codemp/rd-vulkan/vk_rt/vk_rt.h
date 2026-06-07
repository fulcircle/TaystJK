#ifndef VK_RT_H
#define VK_RT_H

// Shared RT light types. Defined here (rather than inside vk_rt.cpp) so that
// world_s in tr_local.h -- which holds the synthesized light list -- can see the
// type. tr_local.h includes this; vk_rt.cpp includes it too.

typedef enum {
	LIGHT_TYPE_POLYGON,
	LIGHT_TYPE_SPHERE,
	LIGHT_TYPE_SPOT
} lightType;

// One emissive triangle = one polygon light. Flat float arrays (not vec3/mat3)
// so the C layout and the GLSL std430 RtLight share the same tight stride on a
// direct upload -- no 16-byte vector padding to reconcile.
typedef struct {
	float     color[3];      // emitted radiance (white * surfaceLight for now)
	float     positions[9];  // v0, v1, v2  (flat 3x vec3)
	float	  normal[3];
	float     lightCentroid[3];
	float	  area;
	float     boundingRadius;
	lightType type;          // LIGHT_TYPE_POLYGON
} rtLight_t;

void R_rtCopyFrameToHistory( void );

#endif // VK_RT_H
