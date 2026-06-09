#ifndef VK_RT_ARENA_H
#define VK_RT_ARENA_H

#include <cstddef>   // size_t
#include <cstdint>   // uint32_t
#include <cstdlib>   // realloc, free

#define ArenaInit(Type) arena_init(sizeof(Type), __alignof(Type))
#define ArenaNext(arena, Type) (Type *)arena_alloc(arena)

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
	size_t   elemSize;
	size_t   align;        // power of two; must divide elemSize for tight packing
	uint32_t numElements;
} arena_t;

static inline arena_t arena_init( size_t elemSize, size_t align ) {
	arena_t a = { 0 };
	a.elemSize = elemSize;
	a.align    = align;
	return a;
}

static inline void *arena_alloc( arena_t *a );
static inline void arena_free( arena_t *a );

// A non-templated C++ class wrapping arena_t that stores type information as
// member variables, with explicit initialization and destruction (no RAII).
class DynamicArray {
public:
	void init(size_t elemSize, size_t align) {
		m_arena = arena_init(elemSize, align);
		m_elemSize = elemSize;
		m_align = align;
	}

	void free() {
		arena_free(&m_arena);
	}

	void* alloc() {
		return arena_alloc(&m_arena);
	}

	void* buffer() const {
		return m_arena.base;
	}

	uint32_t count() const {
		return m_arena.numElements;
	}

	size_t used() const {
		return m_arena.used;
	}

private:
	arena_t m_arena;
	size_t  m_elemSize;
	size_t  m_align;
};

// Returns a pointer to a fresh, uninitialized slot; grows the backing buffer if
// full. The caller fills the returned slot.
static inline void *arena_alloc( arena_t *a ) {
	size_t offset = ( a->used + ( a->align - 1 ) ) & ~( a->align - 1 );
	if ( offset + a->elemSize > a->capacity ) {
		size_t newCap = a->capacity ? a->capacity * 2 : 64 * 1024;
		while ( offset + a->elemSize > newCap )
			newCap *= 2;
		char *grown = (char *)realloc( a->base, newCap );
		if ( !grown )
			ri.Error( ERR_DROP, "arena_alloc: OOM growing to %u bytes", (unsigned)newCap );
		a->base     = grown;
		a->capacity = newCap;
	}
	void *p = a->base + offset;
	a->used = offset + a->elemSize;
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
