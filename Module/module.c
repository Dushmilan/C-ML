#include "module.h"
#include <stdlib.h>   
#include <stdbool.h> 



#include "module.h"

#include <stdbool.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "Autograd.h"
#include "memory_pool.h"
#include "ops.h"
#include "tensor.h"

void module_init(Module *m, ModuleForward fwd, void (*free_self)(Module*)) {
    m->cap_params = 4; m->n_params = 0;
    m->params = (Tensor **)calloc((size_t)m->cap_params, sizeof(Tensor *));
    m->cap_submodules = 4; m->n_submodules = 0;
    m->submodules = (Module **)calloc((size_t)m->cap_submodules, sizeof(Module *));
    m->forward = fwd;
    m->free_self = free_self;
}

int module_add_param(Module *m, Tensor *p) {
    if (m->n_params >= m->cap_params) {
        size_t nc = m->cap_params * 2;
        Tensor **g = (Tensor **)realloc(m->params, nc * sizeof(Tensor *));
        if (!g) return -1;
        m->params = g; m->cap_params = nc;
    }
    p->requires_grad = true;
    m->params[m->n_params++] = p;
    return 0;
}

int module_add_submodule(Module *m, Module *child) {
    if (!m || !child) return -1;
    if (m->n_submodules >= m->cap_submodules) {
        size_t nc = m->cap_submodules * 2;
        Module **g = (Module **)realloc(m->submodules, nc * sizeof(Module *));
        if (!g) return -1;
        m->submodules = g; m->cap_submodules = nc;
    }
    m->submodules[m->n_submodules++] = child;
    return 0;
}

void module_zero_grad(Module *m) {
    if (!m) return;
    for (size_t i = 0; i < m->n_params; i++) {
        Tensor *p = m->params[i];
        if (!p) continue;
        if (p->grad_fn) p->grad_fn->grad = NULL;
        p->grad = NULL;
    }
    for (size_t i = 0; i < m->n_submodules; i++)
        if (m->submodules[i]) module_zero_grad(m->submodules[i]);
}

static void _collect_params(Module *m, Tensor ***out, size_t *n, size_t *cap) {
    for (size_t i = 0; i < m->n_params; i++) {
        if (*n >= *cap) {
            *cap *= 2;
            *out = (Tensor **)realloc(*out, *cap * sizeof(Tensor *));
            if (!*out) return;
        }
        (*out)[(*n)++] = m->params[i];
    }
    for (size_t i = 0; i < m->n_submodules; i++)
        if (m->submodules[i]) _collect_params(m->submodules[i], out, n, cap);
}

void module_parameters(Module *m, Tensor ***out, size_t *n) {
    if (!out || !n) return;
    *out = NULL; *n = 0;
    size_t cap = 8;
    *out = (Tensor **)malloc(cap * sizeof(Tensor *));
    if (!*out) return;
    _collect_params(m, out, n, &cap);
}

Linear *linear_create(size_t in_features, size_t out_features, bool bias) {
    Linear *l = (Linear *)calloc(1, sizeof(Linear));
    if (!l) return NULL;

    module_init(&l->base, linear_forward, (void (*)(Module *))linear_free);
    l->in_features = in_features;
    l->out_features = out_features;
    l->bias = bias;

    size_t wshape[2] = { in_features, out_features };
    Tensor *W_init = tensor_xavier_uniform(wshape, 2);
    l->W = tensor_persistent_create(wshape, 2);
    if (!l->W) { linear_free(l); return NULL; }
    if (W_init) {
        memcpy(l->W->data, W_init->data, in_features * out_features * sizeof(float));
    }

    module_add_param(&l->base, l->W);

    if (bias) {
        size_t bshape[1] = { out_features };
        l->b = tensor_persistent_create(bshape, 1);
        if (!l->b) { linear_free(l); return NULL; }
        module_add_param(&l->base, l->b);
    } else {
        l->b = NULL;
    }
    return l;
}

Tensor *linear_forward(Module *self, const Tensor *x) {
    Linear *l = (Linear *)self;
    if (!x || x->ndim != 2 || x->shape[1] != l->in_features) {
        fprintf(stderr,
                "linear_forward: expected [B, %zu], got %s\n",
                l->in_features,
                x ? (x->ndim == 2 ? "shape[1] mismatch" : "wrong ndim") : "NULL");
        return NULL;
    }

    Tensor *y = matmul(x, l->W);
    if (!y) return NULL;

    if (l->bias && l->b) {
        Tensor *yb = broadcast_add(y, l->b);
        return yb;
    }
    return y;
}

void linear_free(Linear *l) {
    if (!l) return;
    tensor_persistent_free(l->W);
    if (l->b) tensor_persistent_free(l->b);
    free(l->base.params);
    free(l->base.submodules);
    free(l);
}


