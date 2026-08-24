// optimizer.h — SGD optimizer for Phase 2
#ifndef OPTIMIZER_H
#define OPTIMIZER_H

#include "tensor.h"

#include <stddef.h>

typedef struct {
    float lr;
    Tensor **params;
    size_t n_params;
    size_t cap;
} SGD;

SGD *sgd_create(float lr);
int sgd_add_param(SGD *opt, Tensor *param);
void sgd_step(SGD *opt);
void sgd_zero_grad(SGD *opt);
void sgd_free(SGD *opt);

#endif
