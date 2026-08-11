// memory_pool.h
ttypedef struct MemoryPool {
    void* pool;               // Generic pointer
    size_t pool_size;
    size_t used;
    size_t alloc_count;
} MemoryPool;

// Initialize a 1GB pool
MemoryPool* pool_create(size_t bytes);
void* pool_alloc(MemoryPool* pool, size_t bytes);
void pool_reset(MemoryPool* pool);  // Free all at once (per training step)

