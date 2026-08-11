// tensor.h
#ifndef TENSOR_H
#define TENSOR_H

#include <stddef.h>
#include <stdbool.h>

typedef enum {
    FLOAT32,
    FLOAT64,
    INT32,
    INT64
} DType;

typedef struct {
    float* data;        // Heap-allocated contiguous memory
    size_t* shape;       // Dimensions (e.g., [batch, channels, h, w])
    size_t ndim;         // Number of dimensions
    size_t size;         // Total elements (product of shape)
    DType dtype;
    bool requires_grad;  // For autograd
} Tensor;


#endif