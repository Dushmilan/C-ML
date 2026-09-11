# Understanding the C ML Project

This document explains the core components of the C ML (Machine Learning) library.

---

## Tensor

**File:** `tensor.h`, `tensor.c`

A **Tensor** is the fundamental data structure — a multi-dimensional array that holds numerical data (floats). Think of it as the "building block" for all ML operations.

### Structure

```c
typedef struct Tensor {
    float *data;           // Raw numbers stored contiguously in memory
    size_t *shape;         // Dimensions, e.g., [2, 3, 4] means 2 batches × 3 channels × 4
    size_t ndim;           // Number of dimensions (3 in the example above)
    size_t size;           // Total elements (product of shape: 2×3×4 = 24)
    DType dtype;           // Data type (FLOAT32, FLOAT64, INT32, INT64)
    bool requires_grad;    // If true, autograd tracks operations for gradient computation
    Tensor *grad;          // Accumulated gradient (only for leaf tensors)
    OpNode *grad_fn;       // Pointer to the operation that created this tensor
} Tensor;
```

### Key Concepts

| Concept | Description |
|---------|-------------|
| **shape** | Defines tensor dimensions. A 2×3 matrix has shape `[2, 3]` with 6 elements. |
| **data** | Flat array of floats. A 2×3 tensor stores 6 floats sequentially in row-major order. |
| **ndim** | Number of axes/dimensions. A vector is 1D, a matrix is 2D, a color image is 3D (H×W×C). |
| **requires_grad** | When true, this tensor is a "leaf" that accumulates gradients during backpropagation. |
| **grad_fn** | Points to the OpNode that created this tensor (used for autograd graph traversal). |

### Creating Tensors

```c
// Pool-backed (temporary, freed on pool_reset)
size_t shape[2] = {3, 4};
Tensor *t = tensor_create(shape, 2);  // 3×4 matrix, zeros

// Persistent (survives pool_reset, must be explicitly freed)
Tensor *weights = tensor_persistent_create(shape, 2);
tensor_persistent_free(weights);

// Random initialization
Tensor *randn_t = tensor_randn(shape, 2);         // Standard normal ~ N(0,1)
Tensor *xavier = tensor_xavier_uniform(shape, 2);  // Uniform U(-a, a)
```

### Operations

| Function | Description |
|----------|-------------|
| `tensor_create(shape, ndim)` | Allocate and zero-initialize a new tensor |
| `tensor_free(tensor)` | No-op (freed via pool_reset) |
| `tensor_persistent_create(shape, ndim)` | malloc-backed tensor that survives pool reset |
| `tensor_persistent_free(tensor)` | Free persistent tensor memory |
| `tensor_clone(tensor)` | Deep copy — new memory, same values |
| `tensor_fill(tensor, value)` | Set all elements to a scalar |
| `tensor_print(tensor)` | Print shape, dtype, and data |
| `tensor_randn(shape, ndim)` | Fill with standard normal random values |
| `tensor_xavier_uniform(shape, ndim)` | Xavier uniform initialization |
| `tensor_xavier_normal(shape, ndim)` | Xavier normal initialization |
| `tensor_random_seed(seed)` | Set random seed for reproducibility |

---

## Memory Pool

**File:** `memory_pool.h`, `memory_pool.c`

A **Memory Pool** is a fast, simple allocator that pre-allocates one large block of memory and hands out chunks from it. Instead of calling `malloc`/`free` for every tensor (slow), we allocate once and reset in bulk.

### Structure

```c
typedef struct MemoryPool {
    void *pool;        // Pointer to the large memory block (1 GiB by default)
    size_t pool_size;  // Total capacity in bytes
    size_t used;       // Bytes currently allocated
    size_t alloc_count;// Number of allocations made
} MemoryPool;
```

### How It Works

```
┌─────────────────────────────────────────────────────────┐
│  Pool (1 GiB)                                          │
├──────────┬──────────┬──────────┬──────────┬────────────┤
│ Tensor A │ Tensor B │ OpNode   │  ...     │  (free)    │
│ 100 bytes│ 200 bytes│ 80 bytes │          │            │
└──────────┴──────────┴──────────┴──────────┴────────────┘
                              ↑
                          used pointer
```

1. **`pool_alloc(pool, bytes)`** — Bump allocator: moves the `used` pointer forward by `bytes` (with 16-byte alignment) and returns the old position. O(1) — no searching.
2. **`pool_reset(pool)`** — Resets `used` back to 0. All previously allocated memory becomes "free" instantly. This is called once per training step.
3. **No individual free** — You cannot free a single tensor. Everything allocated in a step is freed together via `pool_reset`.

### API

| Function | Description |
|----------|-------------|
| `pool_create(bytes)` | Allocate a pool with `bytes` capacity |
| `pool_alloc(pool, bytes)` | Allocate `bytes` from the pool (returns NULL if full) |
| `pool_reset(pool)` | Reset the pool (free all allocations at once) |
| `get_pool()` | Get the global 1 GiB pool instance |

### Why This Design?

- **Speed**: `malloc` is slow (thread locks, fragmentation). Bump allocation is nearly free.
- **Simplicity**: No need to track individual allocations or worry about leaks.
- **Pattern fit**: ML workloads allocate many small objects per step, then discard them all. Pool reset is perfect for this.

### Persistent vs Pool-Backed

| Type | Allocation | Lifetime | Use Case |
|------|-----------|----------|----------|
| Pool-backed (`tensor_create`) | `pool_alloc` | Current training step | Intermediates, activations |
| Persistent (`tensor_persistent_create`) | `malloc` | Until explicitly freed | Model weights, biases |

---

## Operations (Ops)

**File:** `ops.h`, `ops.c`

**Ops** are the mathematical operations that transform tensors. Each operation takes one or more input tensors and produces a new output tensor.

### Available Operations

| Function | Formula | Description |
|----------|---------|-------------|
| `matmul(A, B)` | `C = A × B` | Matrix multiplication (2D only). Inner dimensions must match. |
| `add(A, B)` | `C = A + B` | Element-wise addition. Both tensors must have the same shape. |
| `broadcast_add(A, B)` | `C = A + B` | Element-wise addition with broadcasting (shapes can differ). |
| `relu(A)` | `C = max(0, A)` | ReLU activation — negative values become 0. |
| `softmax(A, axis)` | `C_i = exp(A_i) / Σ exp(A_j)` | Softmax along specified axis. Outputs sum to 1. |
| `cross_entropy_loss(logits, targets)` | `L = -Σ t_i × log(p_i)` | Classification loss between predictions and targets. |

### Broadcasting

`broadcast_add` allows adding tensors with different shapes by "stretching" dimensions of size 1:

```
A: shape [3, 1]        B: shape [1, 4]        → C: shape [3, 4]
[[1],                   [[10, 20, 30, 40]]      [[11, 21, 31, 41],
 [2],                                            [12, 22, 32, 42],
 [3]]                                            [13, 23, 33, 43]]
```

Rules:
- Dimensions must be equal, or one of them must be 1
- Dimensions are aligned from the right (trailing dimension)

### How Ops Integrate with Autograd

Every operation automatically builds a computational graph:

```c
Tensor *C = matmul(A, B);  // If A or B requires_grad:
                           //   1. Creates an OpNode
                           //   2. Links it to inputs A and B
                           //   3. Stores backward function
                           //   4. Sets C->grad_fn = node
```

When `tensor_backward(C)` is called:
1. Traverse the graph from C backward to leaf tensors
2. Compute gradients using chain rule
3. Accumulate gradients in `tensor->grad` for leaf tensors with `requires_grad = true`

### Shape Examples

```c
// matmul: (m, k) × (k, n) → (m, n)
size_t A_shape[2] = {3, 4};  // 3×4 matrix
size_t B_shape[2] = {4, 2};  // 4×2 matrix
Tensor *C = matmul(A, B);     // Result: 3×2 matrix

// add: same shapes
Tensor *D = add(A, A);        // 3×4 matrix

// relu: preserves shape
Tensor *E = relu(A);          // 3×4 matrix

// softmax: along axis
size_t shape[2] = {2, 5};
Tensor *logits = tensor_create(shape, 2);
Tensor *probs = softmax(logits, 1);  // Softmax over 5 classes (axis=1)
```

---

## Summary: How It All Fits Together

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Memory    │     │   Tensor    │     │    Ops      │
│    Pool     │────▶│   System    │────▶│  (matmul,   │
│             │     │             │     │   add, relu)│
└─────────────┘     └─────────────┘     └─────────────┘
                           │
                           ▼
                    ┌─────────────┐
                    │  Autograd   │
                    │ (backward)  │
                    └─────────────┘
```

1. **Memory Pool** allocates fast — tensors are created from the pool
2. **Tensors** hold data and metadata — they are the input/output of operations
3. **Ops** transform tensors — each op creates new tensors and optionally records an OpNode
4. **Autograd** uses the OpNode graph to compute gradients via backpropagation

This is a minimal but complete ML framework in C, mirroring the design of PyTorch/TensorFlow but without the overhead.
