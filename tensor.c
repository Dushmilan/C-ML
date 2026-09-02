#include "tensor.h"

#include "Autograd.h"
#include "memory_pool.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <time.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846f
#endif

/*
 * tensor_create: allocate and initialize a Tensor on the heap.
 * Parameters:
 *   shape  - array of dimension sizes (e.g., [2, 3, 4])
 *   ndim   - number of dimensions
 * Returns a pointer to a new Tensor, or NULL on failure.
 */

Tensor *tensor_create(size_t *shape, size_t ndim) {
    // Allocate one Tensor struct on the heap
    Tensor *tensor = (Tensor *)pool_alloc(get_pool(), sizeof(Tensor));
    if (!tensor)
        return NULL;

    tensor->ndim = ndim;
    tensor->shape = (size_t *)pool_alloc(get_pool(), ndim * sizeof(size_t));
    if (!tensor->shape)
        return NULL;

    tensor->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    tensor->dtype = FLOAT32;
    tensor->requires_grad = false;
    tensor->grad = NULL;
    tensor->grad_fn = NULL;
    tensor->data = (float *)pool_alloc(get_pool(), tensor->size * sizeof(float));
    if (!tensor->data)
        return NULL;

    memset(tensor->data, 0, tensor->size * sizeof(float)); // replaces calloc
    return tensor;
}

// tensor_free: release all memory owned by the tensor
void tensor_free(Tensor *t) {
    (void)t;
} /* freed wholesale via pool_reset */

// tensor_fill: set every element to the same scalar value

void tensor_fill(Tensor *tensor, float value) {
    if (!tensor || !tensor->data)
        return;
    for (size_t i = 0; i < tensor->size; i++) {
        tensor->data[i] = value;
    }
}
// tensor_print: display shape, dtype, requires_grad, and data

void tensor_print(const Tensor *tensor) {
    if (!tensor) {
        printf("Tensor(NULL)\n");
        return;
    }

    printf("Tensor(shape=[");
    for (size_t i = 0; i < tensor->ndim; i++) {
        printf("%zu", tensor->shape[i]);
        if (i < tensor->ndim - 1)
            printf(", ");
    }
    printf("], dtype=%s, requires_grad=%s)\n", tensor->dtype == FLOAT32 ? "FLOAT32" : "FLOAT64",
           tensor->requires_grad ? "true" : "false");

    if (tensor->size <= 100) {
        printf(" data=[");
        for (size_t i = 0; i < tensor->size; i++) {
            printf("%.4f", tensor->data[i]);
            if (i < tensor->size - 1)
                printf(", ");
        }
        printf("]\n");

    } else {
        printf("  data=[");
        for (size_t i = 0; i < 10; i++)
            printf("%.4f, ", tensor->data[i]);
        printf("... (truncated)]\n");
    }
}

// Persistent tensor: malloc-backed, survives pool_reset

Tensor *tensor_persistent_create(size_t *shape, size_t ndim) {
    Tensor *tensor = (Tensor *)malloc(sizeof(Tensor));
    if (!tensor)
        return NULL;

    tensor->ndim = ndim;
    tensor->shape = (size_t *)malloc(ndim * sizeof(size_t));
    if (!tensor->shape) {
        free(tensor);
        return NULL;
    }

    tensor->size = 1;
    for (size_t i = 0; i < ndim; i++) {
        tensor->shape[i] = shape[i];
        tensor->size *= shape[i];
    }

    tensor->dtype = FLOAT32;
    tensor->requires_grad = false;
    tensor->grad = NULL;
    tensor->grad_fn = NULL;
    tensor->data = (float *)malloc(tensor->size * sizeof(float));
    if (!tensor->data) {
        free(tensor->shape);
        free(tensor);
        return NULL;
    }

    memset(tensor->data, 0, tensor->size * sizeof(float));
    return tensor;
}

void tensor_persistent_free(Tensor *t) {
    if (!t)
        return;
    free(t->data);
    free(t->shape);
    // Note: grad and grad_fn may be pool-allocated — do not free here.
    // Caller should clear grad before free or ensure it was pool-reset.
    free(t);
}

// tensor_clone: deep-copy a tensor (new memory, same values)

Tensor *tensor_clone(const Tensor *tensor) {
    if (!tensor)
        return NULL;

    Tensor *clone = tensor_create(tensor->shape, tensor->ndim);
    if (!clone)
        return NULL;

    clone->dtype = tensor->dtype;
    clone->requires_grad = tensor->requires_grad;
    memcpy(clone->data, tensor->data, tensor->size * sizeof(float));

    return clone;
}


static int have_next_gaussian = 0;
static float next_gaussian;

static float randn(void) {

    if (have_next_gaussian){
        have_next_gaussian = 0;
        return next_gaussian;
    }
    float u1 = ((float)rand() / RAND_MAX);
    float u2 = ((float)rand() / RAND_MAX);
    // Avoid log(0)

    float r = sqrtf(-2.0f * logf(u1<1e-9f ? 1e-9f : u1));
    float theta = 2.0f * (float)M_PI * u2;

    next_gaussian = r * cosf(theta);
    have_next_gaussian = 1;
    return r * sinf(theta);
}

static float rand_uniform(void){
    return (float)rand() / RAND_MAX;
}

Tensor *tensor_randn(size_t *shape, size_t ndim) {
    Tensor *t = tensor_create(shape, ndim);
    if (!t) return NULL;
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = randn();
    return t;
}

Tensor *tensor_xavier_uniform(size_t *shape, size_t ndim) {
    Tensor *t = tensor_create(shape, ndim);
    if (!t || ndim != 2) return t;
    size_t fan_in = shape[1];
    size_t fan_out = shape[0];
    float a = sqrtf(6.0f / ((float)(fan_in + fan_out)));
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = rand_uniform() * 2.0f * a - a; // U(-a, a)
    return t;
}

Tensor *tensor_xavier_normal(size_t *shape, size_t ndim) {
    Tensor *t = tensor_create(shape, ndim);
    if (!t || ndim != 2) return t;
    size_t fan_in = shape[1];
    size_t fan_out = shape[0];
    float std = sqrtf(2.0f / ((float)(fan_in + fan_out)));
    for (size_t i = 0; i < t->size; i++)
        t->data[i] = randn() * std;
    return t;
}

Tensor *tensor_xavier_uniform_fan(size_t fan_in, size_t fan_out) {
    size_t shape[2] = {fan_out, fan_in};
    return tensor_xavier_uniform(shape, 2);
}
void tensor_random_seed(unsigned int seed) {
    srand(seed);
    have_next_gaussian = 0;
}