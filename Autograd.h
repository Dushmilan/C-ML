// Autograd.h
#ifndef AUTOGRAD_H
#define AUTOGRAD_H

#include <stddef.h>
#include "tensor.h"

typedef struct OpNode {
    struct OpNode** inputs;    // Array of input nodes
    size_t n_inputs;
    void (*backward)(struct OpNode*);
    Tensor* grad;              // Gradient of the tensor this node produces
    Tensor* value;             // Tensor produced by this node
    Tensor** saved;            // Array of saved tensors for backward
    size_t n_saved;
    int axis;                  // Operation-specific axis, e.g. softmax
    int visited;               // Temporary topological-sort marker
} OpNode;

OpNode* opnode_create(OpNode** inputs, size_t n_inputs,
                      void (*backward)(OpNode*));
int opnode_save(OpNode* node, Tensor* tensor);
OpNode* node_of(Tensor* tensor);

void autograd_backward_matmul(OpNode* node);
void autograd_backward_add(OpNode* node);
void autograd_backward_relu(OpNode* node);
void autograd_backward_softmax(OpNode* node);
void autograd_backward_cross_entropy(OpNode* node);

void tensor_backward(Tensor* tensor);  // Triggers full backprop

#endif