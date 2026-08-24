#include "optimizer.h"

#include "Autograd.h"

#include <stdlib.h>

SGD *sgd_create(float lr) {
    SGD *opt = (SGD *)malloc(sizeof(SGD));
    if (!opt)
        return NULL;
    opt->lr = lr;
    opt->cap = 8;
    opt->n_params = 0;
    opt->params = (Tensor **)calloc(opt->cap, sizeof(Tensor *));
    if (!opt->params) {
        free(opt);
        return NULL;
    }
    return opt;
}

int sgd_add_param(SGD *opt, Tensor *param) {
    if (!opt || !param)
        return -1;
    param->requires_grad = true;
    if (opt->n_params >= opt->cap) {
        size_t ncap = opt->cap * 2;
        Tensor **grow = (Tensor **)realloc(opt->params, ncap * sizeof(Tensor *));
        if (!grow)
            return -1;
        opt->params = grow;
        opt->cap = ncap;
    }
    opt->params[opt->n_params++] = param;
    return 0;
}

void sgd_step(SGD *opt) {
    if (!opt)
        return;
    for (size_t i = 0; i < opt->n_params; i++) {
        Tensor *p = opt->params[i];
        if (!p || !p->grad || !p->grad->data)
            continue;
        // grad is pool-allocated; param data is persistent (malloc)
        // w -= lr * grad (elementwise)
        for (size_t j = 0; j < p->size; j++) {
            p->data[j] -= opt->lr * p->grad->data[j];
        }
    }
}

void sgd_zero_grad(SGD *opt) {
    if (!opt)
        return;
    for (size_t i = 0; i < opt->n_params; i++) {
        Tensor *p = opt->params[i];
        if (!p)
            continue;
        // grad is pool-allocated — just null the pointer.
        // pool_reset will reclaim the backing memory next.
        // Also clear the OpNode's grad to avoid dangling.
        if (p->grad_fn)
            p->grad_fn->grad = NULL;
        p->grad = NULL;
    }
}

void sgd_free(SGD *opt) {
    if (!opt)
        return;
    free(opt->params);
    free(opt);
}
