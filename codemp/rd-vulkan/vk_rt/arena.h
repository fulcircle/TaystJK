#ifndef VK_RT_ARENA_H
#define VK_RT_ARENA_H

#include <cstddef>   // size_t
#include <cstdint>   // uint32_t
#include <cstdlib>   // realloc, free
// Lightweight growable bump arena sized to a fixed element. The backing buffer
// grows geometrically via realloc.
//
// NOTE: a grow may MOVE base, so do NOT retain a pointer returned by arena_alloc
// across a later arena_alloc -- fill the slot and move on, and read the final
// contiguous array back from arena.base. Call arena_free when done.
//
// For a homogeneous array, pass align = alignof(T): elemSize then stays a multiple
// of align, so slots pack tightly (used == numElements * elemSize, no gaps).
//
// Renderer utility: arena_alloc reports OOM via ri.Error, so include this AFTER
// the headers that bring `ri` / ERR_DROP into scope (tr_local.h).
typedef struct {
	char    *base;
	size_t   used;
	size_t   capacity;
	uint32_t numElements;
} arena_t;

static inline void *arena_alloc( arena_t *a, size_t elemSize, size_t align );
static inline void arena_free( arena_t *a );

#define PushStruct(arena, Type) (Type *)arena_alloc(arena, sizeof(Type), __alignof(Type))
#define GetBuffer(arena, Type)  (Type *)(arena)->base
#define Clear(arena)            arena_free(arena)

// Returns a pointer to a fresh, uninitialized slot; grows the backing buffer if
// full. The caller fills the returned slot.
static inline void *arena_alloc( arena_t *a, size_t size, size_t align ) {
	size_t offset = ( a->used + ( align - 1 ) ) & ~( align - 1 );
	if ( offset + size > a->capacity ) {
		size_t newCap = a->capacity ? a->capacity * 2 : 64 * 1024;
		while ( offset + size > newCap )
			newCap *= 2;
		char *grown = (char *)realloc( a->base, newCap );
		if ( !grown )
			ri.Error( ERR_DROP, "arena_alloc: OOM growing to %u bytes", (unsigned)newCap );
		a->base     = grown;
		a->capacity = newCap;
	}
	void *p = a->base + offset;
	a->used = offset + size;
	a->numElements++;
	return p;
}

// Releases the backing buffer. Safe to call when nothing was allocated
// (free(NULL) is a no-op).
static inline void arena_free( arena_t *a ) {
	free( a->base );
	a->base = NULL;
	a->used = 0;
	a->capacity = 0;
	a->numElements = 0;
}

#endif // VK_RT_ARENA_H
