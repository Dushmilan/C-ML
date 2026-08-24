// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <stdbool.h>
#include <stddef.h>

typedef struct OpNode OpNode;
typedef struct Tensor Tensor;

typedef enum { FLOAT32, FLOAT64, INT32, INT64 } DType;

typedef struct Tensor {
    float *data;   // Heap-allocated contiguous memory
    size_t *shape; // Dimensions (e.g., [batch, channels, h, w])
    size_t ndim;   // Number of dimensions
    size_t size;   // Total elements (product of shape)
    DType dtype;
    bool requires_grad; // For autograd
    Tensor *grad;       // Accumulated gradient (leaf tensors)
    OpNode *grad_fn;    // Operation that produced this tensor
} Tensor;

Tensor *tensor_create(size_t *shape, size_t ndim);
void tensor_free(Tensor *tensor);
void tensor_fill(Tensor *tensor, float value);
void tensor_print(const Tensor *tensor);
Tensor *tensor_clone(const Tensor *tensor);

#endif