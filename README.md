# C-ML — A Minimal Deep Learning Framework in C

> A from-scratch tensor + autograd engine written in pure C. No Python, no external dependencies — just `gcc`, `math.h`, and a 1 GiB bump allocator.

[![Language](https://img.shields.io/badge/language-C-blue.svg)](https://en.wikipedia.org/wiki/C_(programming_language))
[![Build](https://img.shields.io/badge/build-gcc%20--Wall%20--Wextra-success)](Makefile)
[![Format](https://img.shields.io/badge/format-clang--format%2022.1.8%20LLVM-lightgrey)](.clang-format)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](#license)
[![Phase](https://img.shields.io/badge/phase-2%20optimizer%20done-blue)](#roadmap)

---

## Overview

**C-ML** implements the core primitives of modern ML frameworks (PyTorch/JAX) in ~900 lines of C:

- **Tensor** — n-dimensional array with dtype, shape, autograd metadata, and **persistent** (malloc) vs **pool** allocation
- **Memory Pool** — 1 GiB arena / bump allocator for per-step activations (O(1) reclaim via `pool_reset`)
- **Ops** — `matmul`, `add`, `relu`, `softmax`, `cross_entropy_loss` (forward)
- **Autograd** — dynamic computation graph with reverse-mode backprop and topological sort
- **Optimizer** — vanilla **SGD** (`w -= lr * grad`) with `zero_grad` and persistent weight support

Phase 1 is **complete** (tensor/pool/ops/autograd). Phase 2 **optimizer is complete** — training now actually learns. Phase 2 **random initialization is complete** (Xavier/randn). Next: modules, Adam, DataLoader.

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
| **Tensor** | Dynamic `ndim`/`shape`/`size`, `FLOAT32` storage, `requires_grad` + `grad` + `grad_fn`, `tensor_persistent_create` for weights |
| **Memory Pool** | Single 1 GiB backing buffer, 16-byte aligned bump allocator, `alloc_count` tracking, zero per-allocation `free` overhead |
| **Ops (Forward)** | `matmul` (2D, O(m·k·n)), `add` (elementwise), `relu`, `softmax` (stable, arbitrary axis), `cross_entropy_loss` (mean, clamped) |
| **Autograd (Backward)** | Per-op `backward` kernels, gradient accumulation (`_acc_grad`), leaf memoization (`node_of`), recursive post-order DFS (`_topo`), scalar loss seeding |
| **Optimizer** | `SGD` — `sgd_create(lr)`, `sgd_add_param`, `sgd_step(w -= lr*grad)`, `sgd_zero_grad`, `sgd_free`; works with persistent params + pool grads |
| **Tooling** | `Makefile` (`make test` 7 tests), `clang-format` LLVM 100-col, `tests/test_all.c` harness |
| **Zero Dependencies** | Only `libc` + `libm` |

---

## Architecture

```
Main.c (training loop: DataLoader → forward → backward → SGD → reset)
   │
   ├── tensor.h/c ──┐ (pool vs persistent)
   ├── ops.h/c ─────┼──► memory_pool.h/c (arena: pool_create / pool_alloc / pool_reset / get_pool)
   ├── Autograd.h/c ┘ (opnode_create, node_of, _acc_grad, _topo, tensor_backward)
   └── optimizer.h/c    (sgd_create, sgd_add_param, sgd_step, sgd_zero_grad)
```

**Data flow for one step (Phase 2):**

```
w[3,2] persistent (malloc) ──┐
                              ├──► matmul(x,w) → logits[2,2] → softmax → probs → cross_entropy → loss[1]
x[2,3] pool (activations) ────┘          ▲ graph built on requires_grad
                                         │
                                     tensor_backward(loss) → topo → backward → w.grad (pool)
                                         │
                                     sgd_step: w -= lr*w.grad (reads pool grad before reset)
                                     sgd_zero_grad: w.grad=NULL
                                     pool_reset → reclaims x/logits/probs/loss/grads, w persists
```

---

## Project Structure

```
.
├── tensor.h / tensor.c           # Tensor struct, pool create + persistent create, fill/print/clone
├── memory_pool.h / memory_pool.c # Arena allocator (1 GiB, 16B align, bump + reset)
├── ops.h / ops.c                 # Forward ops: matmul, add, relu, softmax, cross_entropy_loss
├── Autograd.h / Autograd.c       # OpNode graph, backward kernels, topo sort, tensor_backward
├── optimizer.h / optimizer.c     # SGD optimizer (vanilla GD, persistent params)
├── Main.c                        # Phase 2 demo: persistent w + SGD, loss 0.346→0.323 in 5 steps
├── Matrix_Basic_Fun.h/c          # Legacy matrix helpers (kept for reference)
├── tests/test_all.c              # 7 tests: create_and_fill, persistent_survives, sgd_step, end_to_end, randn_basic, xavier_uniform_2d, xavier_normal_2d
├── Makefile                      # CC=gcc, CFLAGS=-Wall -Wextra -I., SRC+=optimizer.c, `make test`
├── .clang-format                 # LLVM, IndentWidth 4, ColumnLimit 100, SortIncludes CaseSensitive
└── README.md
```

---

## Quick Start

### Requirements

- `gcc` (≥ 11) with `-lm`
- `clang-format` 22.1.8 (optional) — `pip install clang-format`
- Linux / macOS (tested on Linux)

### Build & Run

```bash
# Phase 2 training demo (SGD, persistent weights)
gcc -Wall -Wextra tensor.c ops.c Autograd.c memory_pool.c optimizer.c Main.c -lm -o /tmp/cml
/tmp/cml
# Initial w: Xavier-uniform initialized
# step 0: loss varies with Xavier init
# step 1: loss decreasing
# Final w: learned weights
echo $?  # 0
```

### Run Tests

```bash
make test
# gcc -Wall -Wextra -I. tests/test_all.c tensor.c ops.c Autograd.c memory_pool.c optimizer.c -lm -o /tmp/test_all && /tmp/test_all
# 7 passed 0 failed
```

### Format

```bash
clang-format -i Main.c tensor.c tensor.h ops.c ops.h Autograd.c Autograd.h memory_pool.c memory_pool.h optimizer.c optimizer.h tests/test_all.c
clang-format --dry-run --Werror Main.c tensor.c ops.c Autograd.c memory_pool.c optimizer.c
```

---

## Usage

### Phase 2 — Training with SGD (Persistent Weights)

```c
#include "Autograd.h"
#include "memory_pool.h"
#include "optimizer.h"
#include "ops.h"
#include "tensor.h"

#define N_STEPS 5

int main() {
    tensor_random_seed(42);
    // Persistent weight — malloc-backed, survives pool_reset
    Tensor *w_init = tensor_xavier_uniform((size_t[]){3, 2}, 2);
    Tensor *w = tensor_persistent_create((size_t[]){3, 2}, 2);
    memcpy(w->data, w_init->data, 3 * 2 * sizeof(float));
    w->requires_grad = true;
    SGD *opt = sgd_create(0.01f);
    sgd_add_param(opt, w); // sets requires_grad=true

    for (int step = 0; step < N_STEPS; step++) {
        Tensor *x = tensor_create((size_t[]){2, 3}, 2);
        x->data[0]=1; x->data[1]=2; x->data[2]=3;
        x->data[3]=4; x->data[4]=5; x->data[5]=6;

        Tensor *targets = tensor_create((size_t[]){2, 2}, 2);
        targets->data[0]=1; targets->data[1]=0;
        targets->data[2]=0; targets->data[3]=1;

        Tensor *logits = matmul(x, w);
        Tensor *probs = softmax(logits, 1);
        Tensor *loss = cross_entropy_loss(probs, targets);

        tensor_backward(loss); // populates w->grad (pool)
        sgd_step(opt);        // w -= lr * grad (reads pool grad)
        sgd_zero_grad(opt);   // w->grad=NULL (pool reclaimed next)
        pool_reset(get_pool()); // reclaims x/logits/probs/loss/grads
    }
    tensor_persistent_free(w);
    sgd_free(opt);
}
```

### Phase 1 — Minimal Forward (Pool Only)

```c
for (int step=0; step<100; step++) {
    Tensor *x = tensor_create((size_t[]){2,3},2);
    Tensor *w = tensor_create((size_t[]){3,2},2); // pool — would be wiped (use persistent for training)
    Tensor *logits = matmul(x,w);
    Tensor *probs = softmax(logits,1);
    pool_reset(get_pool());
}
```

### Autograd Standalone

```c
Tensor *x = tensor_create((size_t[]){2,2},2);
x->requires_grad = true;
Tensor *y = relu(x);
Tensor *loss = cross_entropy_loss(softmax(matmul(y,w),1), targets);
tensor_backward(loss); // x.grad, w.grad now populated
```

---

## Core Concepts

### 1. Memory Pool — Why an Arena?

Per-step `malloc/free` fragments and costs ~100 allocs/step. Pool allocates 1 GiB once (`get_pool()` lazy singleton) and bumps `used` with `16B` alignment `(used+15)&~15`. `pool_reset` sets `used=0` — O(1) vs O(n) frees. `tensor_free` is no-op; `tensor_persistent_create` uses `malloc` for weights that must survive resets.

```c
MemoryPool *p = pool_create(1u << 30); // 1 GiB
void *ptr = pool_alloc(p, bytes);       // aligned bump
pool_reset(p);                          // rewind, keep buffer
Tensor *w = tensor_persistent_create(sh,2); // malloc-backed
```

### 2. Tensor — More Than an Array

```c
typedef struct Tensor {
    float *data;      // contiguous row-major
    size_t *shape;    // e.g. [2,3]
    size_t ndim, size;
    DType dtype;      // FLOAT32
    bool requires_grad;
    Tensor *grad;     // dLoss/dThis (pool-allocated)
    OpNode *grad_fn;
} Tensor;
```

`tensor_create` → 3 pool allocs + `memset` 0. `tensor_persistent_create` → 3 `malloc`s. `tensor_clone` → `memcpy`. `tensor_print` truncates at `size>100`.

### 3. Ops — Forward

Each op validates, pool-allocates `out_data`+`Tensor`+`shape`, computes, then optionally builds graph:

| Op | Formula | Saved for Backward |
|----|---------|-------------------|
| `matmul(A,B)` | `C[i,j]= Σ_p A[i,p]·B[p,j]` | `A, B` |
| `add(A,B)` | `C[i]=A[i]+B[i]` | — (clones grad) |
| `relu(A)` | `max(0, A)` | `A` |
| `softmax(A,axis)` | `exp(x-max)/ Σ exp(x-max)` stable | output `p` + `axis` |
| `cross_entropy(logits,targets)` | `- Σ t·log(clamp(p)) / N` | `logits, targets` |

`requires_grad = A.requires_grad || B.requires_grad`.

### 4. Autograd — Reverse Mode

- **`OpNode`** — `{inputs, n_inputs, backward(*), grad, value, saved[], n_saved, axis, visited}`. `malloc`'d not pooled.
- **`_acc_grad`** — discard if `!input`, steal if `grad==NULL`, else `+=` (multi-path sum).
- **Kernels:** `matmul: gA=g·Bᵀ, gB=Aᵀ·g`; `add: clone`; `relu: g·(x>0)`; `softmax: p·(g-dot)`; `cross_entropy: -t/(p·N)`.
- **`tensor_backward`** — DFS `_topo` (post-order, `cap*=2`), seed `ones`, reverse walk calling `backward`.

### 5. Optimizer — SGD & Persistence

**Problem Phase 1:** `w` was pool-allocated → `pool_reset` wiped it, weights never learned.

**Solution Phase 2:** Split lifetime:
- **Persistent** (`tensor_persistent_create` → `malloc`) — `w`, survives resets
- **Ephemeral** (`tensor_create` → pool) — `x`, `logits`, `probs`, `loss`, `grads`, reclaimed each step

```c
SGD *opt = sgd_create(0.01f); // {lr, params[], n_params, cap}
sgd_add_param(opt, w);        // w->requires_grad=true, params grows via realloc
sgd_step(opt);                // for each p: p->data[j] -= lr * p->grad->data[j]
sgd_zero_grad(opt);           // p->grad=NULL; p->grad_fn->grad=NULL (pool memory freed by next reset)
```

Order matters: `backward → step (reads pool grad) → zero_grad (nulls) → pool_reset (reclaims)`. This is vanilla **(mini-batch) GD** — `B = x->shape[0]` batch size implicit in data, not optimizer. `B=N` → GD, `B=1` → SGD, `B=2` (current) → mini-batch GD.

---

## API Reference

### Tensor (`tensor.h`)

| Function | Signature | Description |
|----------|-----------|-------------|
| `tensor_create` | `Tensor *tensor_create(size_t *shape, size_t ndim)` | Pool-alloc Tensor, copies shape, zeros data. `NULL` on OOM |
| `tensor_persistent_create` | `Tensor *tensor_persistent_create(size_t *shape, size_t ndim)` | `malloc`-backed, survives `pool_reset` |
| `tensor_free` | `void tensor_free(Tensor *t)` | No-op — reclaimed by `pool_reset` |
| `tensor_persistent_free` | `void tensor_persistent_free(Tensor *t)` | `free(data)+free(shape)+free(t)` |
| `tensor_fill` | `void tensor_fill(Tensor *t, float v)` | Fill all `size` elements |
| `tensor_print` | `void tensor_print(const Tensor *t)` | Print shape/dtype/requires_grad + data (trunc>100) |
| `tensor_clone` | `Tensor *tensor_clone(const Tensor *t)` | Deep copy via pool + `memcpy` |
| `tensor_randn` | `Tensor *tensor_randn(size_t *shape, size_t ndim)` | Pool tensor with standard normal values ~N(0,1) |
| `tensor_xavier_uniform` | `Tensor *tensor_xavier_uniform(size_t *shape, size_t ndim)` | Pool tensor with uniform Xavier init U(-a,a) for 2D |
| `tensor_xavier_normal` | `Tensor *tensor_xavier_normal(size_t *shape, size_t ndim)` | Pool tensor with normal Xavier init N(0,std^2) for 2D |
| `tensor_xavier_uniform_fan` | `Tensor *tensor_xavier_uniform_fan(size_t fan_in, size_t fan_out)` | Convenience: Xavier uniform with explicit fan sizes |
| `tensor_random_seed` | `void tensor_random_seed(unsigned int seed)` | Seed RNG for reproducible init |

### Memory Pool (`memory_pool.h`)

| Function | Description |
|----------|-------------|
| `pool_create(bytes)` | `malloc` backing buffer |
| `pool_alloc(pool, bytes)` | 16B-aligned bump, `NULL` on OOM |
| `pool_reset(pool)` | `used=0` rewind, keep buffer |
| `get_pool()` | Lazy singleton `1<<30` |

### Ops (`ops.h`)

```c
Tensor *matmul(const Tensor *A, const Tensor *B);              // 2D only
Tensor *add(const Tensor *A, const Tensor *B);                 // size must match
Tensor *relu(const Tensor *A);
Tensor *softmax(const Tensor *A, int axis);
Tensor *cross_entropy_loss(const Tensor *logits, const Tensor *targets); // returns [1]
```

### Autograd (`Autograd.h`)

```c
OpNode *opnode_create(OpNode **inputs, size_t n, void (*backward)(OpNode*));
int     opnode_save(OpNode *node, Tensor *t);
OpNode *node_of(Tensor *t);
void    tensor_backward(Tensor *loss);
```

### Optimizer (`optimizer.h`)

```c
typedef struct { float lr; Tensor **params; size_t n_params; size_t cap; } SGD;
SGD *sgd_create(float lr);
int   sgd_add_param(SGD *opt, Tensor *param); // sets requires_grad
void  sgd_step(SGD *opt);      // w -= lr * grad (grad is pool, w is persistent)
void  sgd_zero_grad(SGD *opt); // nulls grad pointers before pool_reset
void  sgd_free(SGD *opt);
```

---

## Training Loop & Memory Management

```
Persistent: w (malloc) ─────────────────────────────── lives across steps
Ephemeral:  x/logits/probs/loss/grads (pool) ───► pool_reset per step
```

```
Step:  create pool x → forward → backward → sgd_step (reads pool grad) → sgd_zero_grad → pool_reset
Next:  prior pool Tensor* are DANGLING (still in 1 GiB but overwritten), w persists
```

**Pitfalls:**
- Using `x` after `pool_reset` → silent corruption.
- `w` as pool inside loop → wiped (use `tensor_persistent_create` outside loop).
- Missing `requires_grad=true` or `sgd_add_param` → no grad, `backward` no-op.
- `backward` twice without `zero_grad` → grad accumulates (double).

---

## Testing

```bash
make test
```

- Harness `tests/test_all.c` — `TEST`/`ASSERT`/`ASSERT_CLOSE`, `pool_reset` per test, `passed/failed` summary.
- **7 tests:** `tensor_create_and_fill` (shape/fill), `persistent_survives_pool_reset` (malloc survives pool reset), `sgd_step_basic` (1→0.95 with grad 0.5 lr0.1), `end_to_end_training_step` (full forward→backward→step, loss 0.34, weight moves), `randn_basic` (unit normal stats), `xavier_uniform_2d` (bounded by sqrt(6/(fan_in+fan_out))), `xavier_normal_2d` (std matches sqrt(2/(fan_in+fan_out))).
- Next: `matmul` known answer, `softmax` rows sum 1, numerical grad check, pool OOM/align.

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
- [x] `Autograd.c` — `OpNode` creation and backward graph traversal
- [x] `memory_pool.c` — `pool_create`, `pool_alloc`, `pool_reset`, `get_pool` (1 GiB)
- [x] Integrate pool into `tensor.c`/`ops.c`
- [x] Refactor `Main.c` to `Tensor` API
- [x] Add tests (`make test`)
- [x] Format codebase

**Phase 2 — In Progress**
- [x] **Optimizer** — `SGD` vanilla GD `w -= lr*grad`, `sgd_step`/`sgd_zero_grad` (`optimizer.h/c`)
- [x] **Weight persistence** — `tensor_persistent_create` / `tensor_persistent_free` (malloc vs pool), demo loss `0.346 → 0.323`
- [x] Expand tests to 4 (`persistent`, `sgd_step`, `end_to_end`)
- [x] **Random initialization** — `tensor_randn`, `tensor_xavier_uniform`, `tensor_xavier_normal`, `tensor_xavier_uniform_fan`, `tensor_random_seed` (Box-Muller + reproducibility)
- [x] `Linear` / `Module` abstraction
- [ ] Adam (`m/v` moments), additional ops (`broadcast_add`, `conv2d`), `opnode_free`
- [ ] DataLoader (`batch_size`, `shuffle`) & training on real dataset (MNIST/XOR)

---

## Contributing

1. Branch from `main`, write failing test in `tests/test_all.c`
2. Implement, ensure `make test` (7 passed) and `gcc -Wall -Wextra ... -lm` pass
3. Format: `clang-format -i` (CI checks `--dry-run --Werror`)
4. Commit with conventional message, push, open PR

---

## License

MIT — see `LICENSE` (or inherit repository default). Use freely for learning and research.

---

<p align="center"><em>Built to understand, not just to run.</em></p>
