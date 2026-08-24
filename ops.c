#include "ops.h"

#include "Autograd.h"
#include "memory_pool.h"
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

Tensor *matmul(const Tensor *A, const Tensor *B) {
    if (!A || !B || !A->data || !B->data)
        return NULL;

    // For matrix multiplication, expect 2D tensors: (m, k) * (k, n) -> (m, n)
    if (A->ndim != 2 || B->ndim != 2) {
        fprintf(stderr, "matmul currently only supports 2D tensors\n");
        return NULL;
    }

    size_t m = A->shape[0];
    size_t k = A->shape[1];
    size_t k2 = B->shape[0];
    size_t n = B->shape[1];

    if (k != k2) {
        fprintf(stderr, "matmul dimension mismatch: %zu vs %zu\n", k, k2);
        return NULL;
    }

    size_t total = m * n;
    float *out_data = (float *)pool_alloc(get_pool(), total * sizeof(float));
    if (!out_data)
        return NULL;

    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            float sum = 0.0f;
            for (size_t p = 0; p < k; p++) {
                sum += A->data[i * k + p] * B->data[p * n + j];
            }
            out_data[i * n + j] = sum;
        }
    }

    Tensor *result = (Tensor *)pool_alloc(get_pool(), sizeof(Tensor));
    if (!result)
        return NULL;

    result->data = out_data;
    result->shape = (size_t *)pool_alloc(get_pool(), 2 * sizeof(size_t));
    if (!result->shape)
        return NULL;
    result->shape[0] = m;
    result->shape[1] = n;
    result->ndim = 2;
    result->size = total;
    result->dtype = FLOAT32;
    result->requires_grad = A->requires_grad || B->requires_grad;
    result->grad = NULL;

    if (result->requires_grad) {
        OpNode *in[2] = {node_of((Tensor *)A), node_of((Tensor *)B)};
        OpNode *node = opnode_create(in, 2, autograd_backward_matmul);
        if (node) {
            node->value = result;
            opnode_save(node, (Tensor *)A);
            opnode_save(node, (Tensor *)B);
            result->grad_fn = node;
        }
    }

    return result;
}

Tensor *add(const Tensor *A, const Tensor *B) {
    if (!A || !B || !A->data || !B->data)
        return NULL;

    if (A->size != B->size) {
        fprintf(stderr, "add: tensor size mismatch (%zu vs %zu)\n", A->size, B->size);
        return NULL;
    }

    float *out_data = (float *)pool_alloc(get_pool(), A->size * sizeof(float));
    if (!out_data)
        return NULL;

    for (size_t i = 0; i < A->size; i++) {
        out_data[i] = A->data[i] + B->data[i];
    }
    Tensor *result = (Tensor *)pool_alloc(get_pool(), sizeof(Tensor));
    if (!result)
        return NULL;

    result->data = out_data;
    result->shape = (size_t *)pool_alloc(get_pool(), A->ndim * sizeof(size_t));
    if (!result->shape)
        return NULL;

    for (size_t i = 0; i < A->ndim; i++) {
        result->shape[i] = A->shape[i];
    }
    result->ndim = A->ndim;
    result->size = A->size;
    result->dtype = A->dtype;
    result->requires_grad = A->requires_grad || B->requires_grad;
    result->grad = NULL;

    if (result->requires_grad) {
        OpNode *in[2] = {node_of((Tensor *)A), node_of((Tensor *)B)};
        OpNode *node = opnode_create(in, 2, autograd_backward_add);
        if (node) {
            node->value = result;
            result->grad_fn = node;
        }
    }

    return result;
}

Tensor *relu(const Tensor *A) {
    if (!A || !A->data)
        return NULL;

    float *out_data = (float *)pool_alloc(get_pool(), A->size * sizeof(float));
    if (!out_data)
        return NULL;

    for (size_t i = 0; i < A->size; i++) {
        out_data[i] = A->data[i] > 0.0f ? A->data[i] : 0.0f;
    }

    Tensor *result = (Tensor *)pool_alloc(get_pool(), sizeof(Tensor));
    if (!result)
        return NULL;

    result->data = out_data;
    result->shape = (size_t *)pool_alloc(get_pool(), A->ndim * sizeof(size_t));
    if (!result->shape)
        return NULL;

    for (size_t i = 0; i < A->ndim; i++) {
        result->shape[i] = A->shape[i];
    }
    result->ndim = A->ndim;
    result->size = A->size;
    result->dtype = A->dtype;
    result->requires_grad = A->requires_grad;
    result->grad = NULL;

    if (result->requires_grad) {
        OpNode *node = opnode_create(NULL, 1, autograd_backward_relu);
        if (node) {
            node->value = result;
            node->inputs[0] = node_of((Tensor *)A);
            opnode_save(node, (Tensor *)A);
            result->grad_fn = node;
        }
    }

    return result;
}

Tensor *softmax(const Tensor *A, int axis) {
    if (!A || !A->data || axis < 0 || (size_t)axis >= A->ndim)
        return NULL;

    size_t axis_size = A->shape[axis];
    float *out_data = (float *)pool_alloc(get_pool(), A->size * sizeof(float));
    if (!out_data)
        return NULL;

    // Compute stride for the axis
    size_t stride = 1;
    for (size_t i = (size_t)axis + 1; i < A->ndim; i++) {
        stride *= A->shape[i];
    }

    size_t outer_stride = stride * axis_size;

    for (size_t i = 0; i < A->size; i += outer_stride) {
        // Find max for numerical stability
        float max_val = A->data[i];
        for (size_t j = 1; j < axis_size; j++) {
            if (A->data[i + j] > max_val)
                max_val = A->data[i + j];
        }

        // Compute exp and sum
        float sum = 0.0f;
        for (size_t j = 0; j < axis_size; j++) {
            out_data[i + j] = (float)exp((double)(A->data[i + j] - max_val));
            sum += out_data[i + j];
        }

        // Normalize
        if (sum > 0.0f) {
            for (size_t j = 0; j < axis_size; j++) {
                out_data[i + j] /= sum;
            }
        }
    }

    Tensor *result = (Tensor *)pool_alloc(get_pool(), sizeof(Tensor));
    if (!result)
        return NULL;

    result->data = out_data;
    result->shape = (size_t *)pool_alloc(get_pool(), A->ndim * sizeof(size_t));
    if (!result->shape)
        return NULL;

    for (size_t i = 0; i < A->ndim; i++) {
        result->shape[i] = A->shape[i];
    }
    result->ndim = A->ndim;
    result->size = A->size;
    result->dtype = FLOAT32;
    result->requires_grad = A->requires_grad;
    result->grad = NULL;

    if (result->requires_grad) {
        OpNode *node = opnode_create(NULL, 1, autograd_backward_softmax);
        if (node) {
            node->value = result;
            node->axis = axis;
            node->inputs[0] = node_of((Tensor *)A);
            opnode_save(node, result);
            result->grad_fn = node;
        }
    }

    return result;
}

Tensor *cross_entropy_loss(const Tensor *logits, const Tensor *targets) {
    if (!logits || !targets || !logits->data || !targets->data)
        return NULL;
    if (logits->size != targets->size)
        return NULL;

    float *out_data = (float *)pool_alloc(get_pool(), sizeof(float));
    if (!out_data)
        return NULL;

    float loss = 0.0f;
    for (size_t i = 0; i < logits->size; i++) {
        float p = logits->data[i];
        // Clamp to avoid log(0)
        if (p < 1e-7f)
            p = 1e-7f;
        if (p > 1.0f - 1e-7f)
            p = 1.0f - 1e-7f;
        loss -= targets->data[i] * logf((double)p);
    }
    out_data[0] = loss / (float)logits->size;

    Tensor *result = (Tensor *)pool_alloc(get_pool(), sizeof(Tensor));
    if (!result)
        return NULL;

    result->data = out_data;
    result->shape = (size_t *)pool_alloc(get_pool(), sizeof(size_t));
    if (!result->shape)
        return NULL;
    result->shape[0] = 1;
    result->ndim = 1;
    result->size = 1;
    result->dtype = FLOAT32;
    result->requires_grad = logits->requires_grad || targets->requires_grad;
    result->grad = NULL;

    if (result->requires_grad) {
        OpNode *in[2] = {node_of((Tensor *)logits), node_of((Tensor *)targets)};
        OpNode *node = opnode_create(in, 2, autograd_backward_cross_entropy);
        if (node) {
            node->value = result;
            opnode_save(node, (Tensor *)logits);
            opnode_save(node, (Tensor *)targets);
            result->grad_fn = node;
        }
    }

    return result;
}