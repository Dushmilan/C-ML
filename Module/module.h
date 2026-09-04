#ifndef MODULE_H
#define MODULE_H

#include <stddef.h>
#include <stdbool.h>
#include "tensor.h"

typedef struct Module Module;

typedef Tensor *(*ModuleForward)(Module *self, const Tensor *x);

typedef struct Module {
    Tensor **params;        // persistent params owned/registered by this module
    size_t   n_params;
    size_t   cap_params;

    Module **submodules;    // optional, for future Sequential/MLP
    size_t   n_submodules;
    size_t   cap_submodules;

    ModuleForward forward;  // v-table
    void (*free_self)(Module *self); // frees the module-specific data
} Module;

// Base API
void module_init(Module *m, ModuleForward fwd, void (*free_self)(Module*));
int  module_add_param(Module *m, Tensor *p);  // sets requires_grad=true
int  module_add_submodule(Module *m, Module *child);
void module_zero_grad(Module *m);  // null p->grad and p->grad_fn->grad

// Collect all params into a flat array (for the optimizer)
void module_parameters(Module *m, Tensor ***out, size_t *n);

// Linear: y = x @ W + b   (W: [in, out], b: [out])
typedef struct {
    Module base;
    size_t in_features;
    size_t out_features;
    bool   bias;
    Tensor *W;   // persistent, Xavier-uniform
    Tensor *b;   // persistent, zeros (or NULL)
} Linear;

Linear *linear_create(size_t in_features, size_t out_features, bool bias);
Tensor *linear_forward(Module *self, const Tensor *x);
void    linear_free(Linear *l);

#endif