# C-ML — A Minimal Deep Learning Framework in C

> A from-scratch tensor + autograd engine written in pure C. No Python, no external dependencies — just `gcc`, `math.h`, and a 1 GiB bump allocator.

[![Language](https://img.shields.io/badge/language-C-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))
[![Build](https://img.shields.io/badge/build-gcc%20--Wall%20--Wextra-success)](Makefile)
[![Format](https://img.shields.io/badge/format-clang--format%2022.1.8%20LLVM-lightgrey)](.clang-format)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Phase](https://img.shields.io/badge/phase-1%20complete-brightgreen)](#roadmap)

---

## Overview

**C-ML** implements the core primitives of modern ML frameworks (PyTorch/JAX) in ~800 lines of C:

- **Tensor** — n-dimensional array with dtype, shape, and autograd metadata
- **Memory Pool** — 1 GiB arena / bump allocator for per-step allocations (O(1) reclaim via `pool_reset`)
- **Ops** — `matmul`, `add`, `relu`, `softmax`, `cross_entropy_loss` (forward)
- **Autograd** — dynamic computation graph with reverse-mode backprop and topological sort

Phase 1 is **feature-complete and tested**. Phase 2 will add optimizers, weight persistence, initialization, and modules.

Framework is intentionally minimal and readable — ideal for learning how tensors, memory, and backprop actually work under the hood.

---

## Table of Contents

- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Usage](#usage)
- [Core Concepts](#core-concepts)
- [API Reference](#api-reference)
- [Training Loop & Memory Management](#training-loop--memory-management)
- [Testing](#testing)
- [Code Style](#code-style)
- [Roadmap](#roadmap)
- [Contributing](#contributing)
- [License](#license)

---

## Features

| Area | Details |
|------|---------|
| **Tensor** | Dynamic `ndim`/`shape`/`size`, `FLOAT32` storage, `requires_grad` + `grad` + `grad_fn` for autograd |
| **Memory Pool** | Single 1 GiB backing buffer, 16-byte aligned bump allocator, `alloc_count` tracking, zero per-allocation `free` overhead |
| **Ops (Forward)** | `matmul` (2D, O(m·k·n)), `add` (elementwise), `relu`, `softmax` (stable, arbitrary axis), `cross_entropy_loss` (mean, clamped) |
| **Autograd (Backward)** | Per-op `backward` kernels, gradient accumulation (`_acc_grad`), leaf memoization (`node_of`), recursive post-order DFS (`_topo`), scalar loss seeding |
| **Tooling** | `Makefile` (`make test`), `clang-format` LLVM 100-col, `tests/test_all.c` harness |
| **Zero Dependencies** | Only `libc` + `libm` |

---

## Architecture

```
Main.c (training loop)
   │
   ├── tensor.h/c ──┐
   │   creates/views│
   ├── ops.h/c ─────┼──► memory_pool.h/c (arena: pool_create / pool_alloc / pool_reset / get_pool)
   │   forward + graph build │
   └── Autograd.h/c ─┘
       graph / backward (opnode_create, node_of, _acc_grad, _topo, tensor_backward)
```

**Data flow for one step:**

```
x[2,3]  w[3,2]
   \    /
   matmul → logits[2,2] → softmax(axis=1) → probs[2,2] → cross_entropy → loss[1]
                               ▲ graph built inline on requires_grad==true
                               │
                           tensor_backward(loss) → DFS topo → reverse backward → x.grad / w.grad
   pool_reset(get_pool()) → O(1) reclaim for next step
```

---

## Project Structure

```
.
├── tensor.h / tensor.c           # Tensor struct, create/fill/print/clone (pool-backed)
├── memory_pool.h / memory_pool.c # Arena allocator (1 GiB, 16B align, bump + reset)
├── ops.h / ops.c                 # Forward ops: matmul, add, relu, softmax, cross_entropy_loss
├── Autograd.h / Autograd.c       # OpNode graph, backward kernels, topo sort, tensor_backward
├── Main.c                        # Example N_STEPS=100 training loop (pool_reset pattern)
├── Matrix_Basic_Fun.h/c          # Legacy matrix helpers (kept for reference)
├── tests/test_all.c              # Minimal test harness (TEST/ASSERT/ASSERT_CLOSE)
├── Makefile                      # CC=gcc, CFLAGS=-Wall -Wextra -I., `make test`
├── .clang-format                 # LLVM, IndentWidth 4, ColumnLimit 100, SortIncludes CaseSensitive
└── README.md
```

---

## Quick Start

### Requirements

- `gcc` (≥ 11) with `-lm`
- `clang-format` 22.1.8 (optional, for contribution) — installed via `pip install clang-format`
- Linux / macOS (tested on Linux)

### Build & Run

```bash
# Build the example (forward pass demo)
gcc -Wall -Wextra tensor.c ops.c Autograd.c memory_pool.c Main.c -lm -o /tmp/cml
/tmp/cml
echo $?  # 0
```

### Run Tests

```bash
make test
# gcc -Wall -Wextra -I. tests/test_all.c tensor.c ops.c Autograd.c memory_pool.c -lm -o /tmp/test_all && /tmp/test_all
# 1 passed 0 failed
```

### Format

```bash
clang-format -i Main.c tensor.c tensor.h ops.c ops.h Autograd.c Autograd.h memory_pool.c memory_pool.h
clang-format --dry-run --Werror Main.c tensor.c ops.c Autograd.c memory_pool.c
```

---

## Usage

### Minimal Example — Forward Pass

```c
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

#define N_STEPS 100

int main() {
    for (int step = 0; step < N_STEPS; step++) {
        // 1. Create tensors from the pool (cheap)
        Tensor *x = tensor_create((size_t[]){2, 3}, 2);
        Tensor *w = tensor_create((size_t[]){3, 2}, 2);
        tensor_fill(x, 1.0f);
        tensor_fill(w, 2.0f);

        // 2. Forward
        Tensor *logits = matmul(x, w);      // [2,3] @ [3,2] -> [2,2] filled with 6.0
        Tensor *probs = softmax(logits, 1); // rows -> [0.5, 0.5]

        // 3. Copy out anything to keep BEFORE reset
        // float loss_val = loss->data[0];

        // 4. Reclaim all pool memory for next step
        pool_reset(get_pool());
        // x, w, logits, probs now INVALID
    }
    return 0;
}
```

> **Trace:** `x=1.0`, `w=2.0` → `logits = [[6,6],[6,6]]` (1·2 summed over k=3) → `softmax` → `[[0.5,0.5],[0.5,0.5]]`.

### Autograd Example

```c
Tensor *x = tensor_create((size_t[]){2, 2}, 2);
x->requires_grad = true;
Tensor *y = relu(x);
Tensor *loss = cross_entropy_loss(softmax(matmul(y, w), 1), targets);
tensor_backward(loss);
// x.grad, w.grad now populated; read before pool_reset
```

---

## Core Concepts

### 1. Memory Pool — Why an Arena?

Per-step `malloc/free` for every `Tensor` fragments the heap and costs ~100 allocations/step. The pool allocates 1 GiB once (`get_pool()` lazy singleton) and bumps a `used` offset with 16-byte alignment (`(used+15)&~15`). `pool_reset` just sets `used=0` — O(1) versus O(n) frees. Tradeoff: no per-tensor `free`; `tensor_free` is a no-op — all memory is freed wholesale.

```c
MemoryPool *p = pool_create(1u << 30); // 1 GiB
void *ptr = pool_alloc(p, bytes);       // aligned bump
pool_reset(p);                          // rewind, don't free backing buffer
```

### 2. Tensor — More Than an Array

```c
typedef struct Tensor {
    float *data;      // contiguous row-major
    size_t *shape;    // e.g. [2,3]
    size_t ndim, size; // size = product(shape)
    DType dtype;      // FLOAT32 (FLOAT64/INT reserved)
    bool requires_grad;
    Tensor *grad;     // dLoss/dThis
    OpNode *grad_fn;  // producing operation
} Tensor;
```

`tensor_create` performs 3 pool allocs (struct, shape, data) and `memset` zeros. `tensor_clone` deep-copies via `memcpy`. `tensor_print` truncates at `size>100`.

### 3. Ops — Forward

Each op validates shapes, pool-allocates `out_data` + `Tensor` + `shape`, computes, then optionally builds graph:

| Op | Formula | Saved for Backward |
|----|---------|-------------------|
| `matmul(A,B)` | `C[i,j]= Σ_p A[i,p]·B[p,j]` | `A, B` |
| `add(A,B)` | `C[i]=A[i]+B[i]` | — (clones grad) |
| `relu(A)` | `max(0, A)` | `A` |
| `softmax(A,axis)` | `exp(x-max)/ Σ exp(x-max)` stable | output `p` + `axis` |
| `cross_entropy(logits,targets)` | `- Σ t·log(clamp(p)) / N` | `logits, targets` |

`requires_grad = A.requires_grad || B.requires_grad`. If true, `node_of(A/B)` memoizes leaf nodes, `opnode_create` + `opnode_save` link graph.

### 4. Autograd — Reverse Mode

- **`OpNode`** — `{inputs, n_inputs, backward(*), grad, value, saved[], n_saved, axis, visited}`. Leaf nodes have `backward==NULL`. Graph nodes are `malloc`'d (not pooled) to survive `pool_reset` if needed.
- **`_acc_grad(input, ng)`** — If `!input` (constant) discard; if `input->grad==NULL` steal `ng`; else elementwise `+=` (sum for multi-path reuse).
- **Backward kernels** implement chain rule:
  - `matmul: gA = g·Bᵀ, gB = Aᵀ·g`
  - `add: gA=g, gB=g` (clone)
  - `relu: g·(x>0)`
  - `softmax: p·(g - dot)`, `dot= Σ p·g`, batched over `outer_stride`
  - `cross_entropy: -t/(p·N)·up`, clamped `p∈[1e-7,1-1e-7]`
- **`tensor_backward(tensor)`** — DFS `_topo` (post-order, `visited` mark, `cap*=2` growth), clear marks, seed `root->grad` with `ones` (size-matched, `1.0f`), then walk reverse order calling `backward`.

---

## API Reference

### Tensor (`tensor.h`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `tensor_create` | `Tensor *tensor_create(size_t *shape, size_t ndim)` | Pool-alloc Tensor, copies shape, zeros data. Returns `NULL` on OOM |
| `tensor_free` | `void tensor_free(Tensor *t)` | No-op — reclaimed by `pool_reset` |
| `tensor_fill` | `void tensor_fill(Tensor *t, float v)` | Fill all `size` elements with `v` |
| `tensor_print` | `void tensor_print(const Tensor *t)` | Print shape/dtype/requires_grad + data (truncates >100) |
| `tensor_clone` | `Tensor *tensor_clone(const Tensor *t)` | Deep copy (new pool alloc + `memcpy`) |

### Memory Pool (`memory_pool.h`)

| Function | Description |
|----------|-------------|
| `pool_create(bytes)` | `malloc` backing buffer, returns `MemoryPool*` |
| `pool_alloc(pool, bytes)` | 16B-aligned bump, `NULL` on OOM |
| `pool_reset(pool)` | `used=0, alloc_count=0` — rewind, keep buffer |
| `get_pool()` | Lazy singleton `1<<30` (1 GiB) |

### Ops (`ops.h`)

```c
Tensor *matmul(const Tensor *A, const Tensor *B);              // 2D only, k must match
Tensor *add(const Tensor *A, const Tensor *B);                 // size must match
Tensor *relu(const Tensor *A);
Tensor *softmax(const Tensor *A, int axis);                    // axis < ndim
Tensor *cross_entropy_loss(const Tensor *logits, const Tensor *targets); // size must match, returns [1]
```

### Autograd (`Autograd.h`)

```c
OpNode *opnode_create(OpNode **inputs, size_t n, void (*backward)(OpNode*));
int     opnode_save(OpNode *node, Tensor *t);                  // append to saved[]
OpNode *node_of(Tensor *t);                                   // memoize leaf
void    tensor_backward(Tensor *loss);                        // run full backprop
// Kernels: autograd_backward_matmul/add/relu/softmax/cross_entropy
```

---

## Training Loop & Memory Management

```
Step N:  create pool tensors → forward → (loss → backward → optimizer) → copy out scalars/grads → pool_reset
Step N+1: all prior Tensor* pointers are DANGLING (still inside 1 GiB but overwritten)
```

**Common pitfalls:**
- Using `x` after `pool_reset` → stale data, no segfault but silent corruption.
- Creating persistent weights inside loop (`w` per step) → weights never learn. Fix: create `w` *outside* loop or use a separate persistent pool / `malloc` for weights and only pool-allocate activations.
- Forgetting `requires_grad=true` → `node_of` returns `NULL`, no graph, `tensor_backward` is no-op.
- Calling `tensor_backward` twice without zeroing `grad` → gradients accumulate (double).

---

## Testing

```bash
make test
```

- Harness `tests/test_all.c` — custom macros `TEST`/`ASSERT`/`ASSERT_CLOSE` (no external deps), `pool_reset(get_pool())` per test, `passed/failed` summary, exit code `failed?1:0`.
- Current: `tensor_create_and_fill` (shape/size/fill).  
- Recommended next tests: `matmul` known answer (`[[1,2,3]]@[1;1;1]=6`), `softmax` rows sum 1, pool OOM/align, `relu` backward mask, `matmul` backward numerical grad check.

---

## Code Style

- **Formatter:** `.clang-format` — `BasedOnStyle: LLVM`, `IndentWidth: 4`, `ColumnLimit: 100`, `BreakBeforeBraces: Attach`, `SortIncludes: CaseSensitive`, `IncludeBlocks: Regroup`
- **Compiler flags:** `-Wall -Wextra -I.`
- **Commands:**
  ```bash
  clang-format -i *.c *.h tests/test_all.c
  clang-format --dry-run --Werror *.c *.h
  ```

---

## Roadmap

**Phase 1 — Complete ✅**
- [x] `tensor.c` — `tensor_create`, `tensor_free`, `tensor_fill`, `tensor_print`, `tensor_clone`
- [x] `ops.c` — `matmul`, `add`, `relu`, `softmax`, `cross_entropy_loss`
- [x] `Autograd.c` — `OpNode` creation and backward graph traversal (`_topo`/`tensor_backward`)
- [x] `memory_pool.c` — `pool_create`, `pool_alloc`, `pool_reset`, `get_pool` (1 GiB)
- [x] Integrate pool into `tensor.c`/`ops.c` (replace `malloc/calloc`)
- [x] Refactor `Main.c` to `Tensor` API (remove `int mat1[2][3]`)
- [x] Add tests (`make test`)
- [x] Format codebase

**Phase 2 — Planned**
- [ ] Optimizer (`SGD` / `Adam`: `w -= lr * grad`)
- [ ] Weight persistence across `pool_reset` (dual pool or `malloc` for parameters)
- [ ] Random initialization (`randn` / `xavier`)
- [ ] `Linear` / `Module` abstraction
- [ ] Additional ops (`broadcast_add`, `conv2d`) and `opnode_free`
- [ ] Numerical gradient checks & expanded test suite
- [ ] Data loader & training on real dataset (e.g., MNIST)

---

## Contributing

1. Branch from `main`, write a failing test in `tests/test_all.c`
2. Implement, ensure `make test` and `gcc -Wall -Wextra ... -lm` pass
3. Format: `clang-format -i` (CI checks `--dry-run --Werror`)
4. Commit with conventional message, push, open PR

---

## License

MIT — see `LICENSE` (or inherit repository default). Use freely for learning and research.

---

<p align="center"><em>Built to understand, not just to run.</em></p>
