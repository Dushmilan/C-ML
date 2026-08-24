#include "memory_pool.h"

#include <stdlib.h>
#include <string.h>

/*
 * pool_create: allocate a fixed-size backing buffer once.
 *   bytes - total capacity of the pool
 * Returns a MemoryPool*, or NULL on failure.
 */

MemoryPool *pool_create(size_t bytes) {
    if (bytes == 0)
        return NULL;

    MemoryPool *p = (MemoryPool *)malloc(sizeof(MemoryPool));
    if (!p)
        return NULL;

    p->pool = malloc(bytes);
    if (!p->pool) {
        free(p);
        return NULL;
    }

    p->pool_size = bytes;
    p->used = 0;
    p->alloc_count = 0;
    return p;
}

/*
 * pool_alloc: bump allocator. Returns the next free region, or NULL if full.
 * 16-byte alignment keeps float/struct accesses safe.
 */

void *pool_alloc(MemoryPool *pool, size_t bytes) {
    if (!pool || !pool->pool || bytes == 0)
        return NULL;

    const size_t align = 16;
    size_t aligned = (pool->used + (align - 1)) & ~(align - 1);

    if (aligned + bytes > pool->pool_size)
        return NULL; // out of memory

    void *ptr = (char *)pool->pool + aligned;
    pool->used = aligned + bytes;
    pool->alloc_count++;
    return ptr;
}

/*
 * pool_reset: free everything at once (e.g. per training step).
 * Does NOT call free() on the backing buffer — just rewinds the pointer.
 */

void pool_reset(MemoryPool *pool) {
    if (!pool)
        return;
    pool->used = 0;
    pool->alloc_count = 0;
    /* Optional: zero memory to avoid stale data. Skip for speed in hot loops.
       if (pool->pool) memset(pool->pool, 0, pool->pool_size); */
}

static MemoryPool *g_pool = NULL;
MemoryPool *get_pool(void) {
    if (!g_pool)
        g_pool = pool_create(1u << 30); // 1 GiB pool for all tensors
    return g_pool;
}
