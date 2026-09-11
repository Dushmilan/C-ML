#include "Autograd.h"

#include "ops.h"
#include "tensor.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/*
 * Accumulate a newly computed gradient tensor into an input node's gradient.
 * If the input has no gradient yet, take ownership of the new one. Otherwise
 * add elementwise (a node reachable from multiple paths sums its gradients)
 * and free the freshly computed tensor.
 */
static void _acc_grad(OpNode *input, Tensor *ng) {
    if (!input) { // Constant input (does not require grad)
        tensor_free(ng);
        return;
    }

    Tensor *g = input->grad;
    if (!g) {
        input->grad = ng;
        if (input->value)
            input->value->grad = ng;
        return;
    }

    for (size_t i = 0; i < g->size; i++)
        g->data[i] += ng->data[i];
    tensor_free(ng);
    if (input->value)
        input->value->grad = g;
}

/*
 * Create a node standing for the given tensor in the graph. Leaf tensors
 * (requires_grad but produced by nothing) get a node with backward == NULL.
 */
OpNode *node_of(Tensor *t) {
    if (!t || !t->requires_grad)
        return NULL;
    if (t->grad_fn)
        return t->grad_fn;
    OpNode *n = opnode_create(NULL, 0, NULL);
    n->value = t;
    t->grad_fn = n;
    return n;
}

OpNode *opnode_create(OpNode **inputs, size_t n_inputs, void (*backward)(OpNode *)) {
    OpNode *node = (OpNode *)malloc(sizeof(OpNode));
    if (!node)
        return NULL;

    node->inputs = NULL;
    node->n_inputs = n_inputs;
    if (n_inputs > 0) {
        node->inputs = (OpNode **)calloc(n_inputs, sizeof(OpNode *));
        if (!node->inputs) {
            free(node);
            return NULL;
        }
        for (size_t i = 0; i < n_inputs; i++)
            node->inputs[i] = inputs ? inputs[i] : NULL;
    }

    node->backward = backward;
    node->grad = NULL;
    node->value = NULL;
    node->saved = NULL;
    node->n_saved = 0;
    node->axis = 0;
    node->visited = 0;
    return node;
}

int opnode_save(OpNode *node, Tensor *tensor) {
    Tensor **ns = (Tensor **)realloc(node->saved, (node->n_saved + 1) * sizeof(Tensor *));
    if (!ns)
        return -1;
    node->saved = ns;
    node->saved[node->n_saved++] = tensor;
    return 0;
}

/* ---------- Operation backward passes ---------- */

/* C = A @ B ; gA = g @ B^T ; gB = A^T @ g */
void autograd_backward_matmul(OpNode *node) {
    Tensor *A = node->saved[0];
    Tensor *B = node->saved[1];
    Tensor *g = node->grad;

    size_t m = A->shape[0], k = A->shape[1];
    size_t n = B->shape[1];

    size_t shA[2] = {m, k};
    size_t shB[2] = {k, n};
    Tensor *gA = tensor_create(shA, 2);
    Tensor *gB = tensor_create(shB, 2);

    for (size_t i = 0; i < m; i++)
        for (size_t p = 0; p < k; p++)
            for (size_t j = 0; j < n; j++)
                gA->data[i * k + p] += g->data[i * n + j] * B->data[p * n + j];

    for (size_t p = 0; p < k; p++)
        for (size_t j = 0; j < n; j++)
            for (size_t i = 0; i < m; i++)
                gB->data[p * n + j] += A->data[i * k + p] * g->data[i * n + j];

    _acc_grad(node->inputs[0], gA);
    _acc_grad(node->inputs[1], gB);
}

/* C = A + B ; gradient flows to both operands unchanged */
void autograd_backward_add(OpNode *node) {
    Tensor *g = node->grad;
    Tensor *gA = tensor_clone(g);
    Tensor *gB = tensor_clone(g);
    if (!gA || !gB) {
        tensor_free(gA);
        tensor_free(gB);
        return;
    }
    _acc_grad(node->inputs[0], gA);
    _acc_grad(node->inputs[1], gB);
}

/* y = relu(x) ; dy/dx = 1 if x > 0 else 0 */
void autograd_backward_relu(OpNode *node) {
    Tensor *A = node->saved[0];
    Tensor *g = node->grad;

    Tensor *gi = tensor_create(A->shape, A->ndim);
    if (!gi)
        return;
    for (size_t i = 0; i < A->size; i++)
        gi->data[i] = g->data[i] * (A->data[i] > 0.0f ? 1.0f : 0.0f);

    _acc_grad(node->inputs[0], gi);
}

/* y = softmax(x) ; dy_i = p_i * (g_i - sum_j p_j * g_j) along the given axis */
void autograd_backward_softmax(OpNode *node) {
    Tensor *p = node->saved[0]; // Softmax output and input share shape
    Tensor *g = node->grad;

    size_t axis = (size_t)node->axis;
    size_t axis_size = p->shape[axis];

    size_t stride = 1;
    for (size_t i = axis + 1; i < p->ndim; i++)
        stride *= p->shape[i];
    size_t outer_stride = stride * axis_size;

    Tensor *gi = tensor_create(p->shape, p->ndim);
    if (!gi)
        return;

    for (size_t i = 0; i < p->size; i += outer_stride) {
        float dot = 0.0f;
        for (size_t j = 0; j < axis_size; j++)
            dot += g->data[i + j] * p->data[i + j];
        for (size_t j = 0; j < axis_size; j++)
            gi->data[i + j] = p->data[i + j] * (g->data[i + j] - dot);
    }

    _acc_grad(node->inputs[0], gi);
}

/* loss = -sum(t * log(clamp(p))) / N ; dL/dp_i = -t_i / (p_i * N) */
void autograd_backward_cross_entropy(OpNode *node) {
    Tensor *logits = node->saved[0];
    Tensor *targets = node->saved[1];
    Tensor *g = node->grad;

    float N = (float)logits->size;
    float up = (g && g->size > 0) ? g->data[0] : 1.0f;

    Tensor *gi = tensor_create(logits->shape, logits->ndim);
    if (!gi)
        return;

    for (size_t i = 0; i < logits->size; i++) {
        float p = logits->data[i];
        if (p < 1e-7f)
            p = 1e-7f;
        if (p > 1.0f - 1e-7f)
            p = 1.0f - 1e-7f;
        gi->data[i] = -targets->data[i] / (p * N) * up;
    }

    _acc_grad(node->inputs[0], gi);
}

void autograd_backward_broadcast_add(OpNode *node) {
    Tensor *A = node->saved[0];
    Tensor *B = node->saved[1];
    Tensor *g = node->grad;
    if (!g) return;

    size_t out_ndim = node->bcast_ndim;
    size_t *sout  = node->bcast_shape;
    size_t *stra  = node->bcast_stra;
    size_t *strb  = node->bcast_strb;

    size_t out_size = 1;
    for (size_t i = 0; i < out_ndim; i++) out_size *= sout[i];

    size_t stride_out[8];
    stride_out[out_ndim - 1] = 1;
    for (size_t i = out_ndim - 1; i-- > 0;)
        stride_out[i] = stride_out[i + 1] * sout[i + 1];

    Tensor *gA = NULL, *gB = NULL;
    if (A->requires_grad)
        gA = tensor_create(A->shape, A->ndim);
    if (B->requires_grad)
        gB = tensor_create(B->shape, B->ndim);
    if (!gA && !gB) return;

    for (size_t gi = 0; gi < out_size; gi++) {
        size_t ai = 0, bi = 0, tmp = gi;
        for (size_t d = 0; d < out_ndim; d++) {
            size_t coord = tmp / stride_out[d];
            tmp %= stride_out[d];
            ai += coord * stra[d];
            bi += coord * strb[d];
        }
        if (gA) gA->data[ai] += g->data[gi];
        if (gB) gB->data[bi] += g->data[gi];
    }

    _acc_grad(node->inputs[0], gA);
    _acc_grad(node->inputs[1], gB);
}

/* ---------- Graph traversal ---------- */

/* Recursive post-order DFS: dependencies appear before dependents. */
static void _topo(OpNode *node, OpNode ***order, size_t *len, size_t *cap) {
    if (!node || node->visited)
        return;
    node->visited = 1;

    for (size_t i = 0; i < node->n_inputs; i++)
        _topo(node->inputs[i], order, len, cap);

    if (*len >= *cap) {
        *cap *= 2;
        OpNode **grow = (OpNode **)realloc(*order, *cap * sizeof(OpNode *));
        if (!grow)
            return;
        *order = grow;
    }
    (*order)[(*len)++] = node;
}

void tensor_backward(Tensor *tensor) {
    if (!tensor || !tensor->requires_grad || !tensor->grad_fn)
        return;

    size_t cap = 16, len = 0;
    OpNode **order = (OpNode **)malloc(cap * sizeof(OpNode *));
    if (!order)
        return;

    _topo(tensor->grad_fn, &order, &len, &cap);
    for (size_t i = 0; i < len; i++)
        order[i]->visited = 0;

    // Seed the root gradient (ones_like): d( sum(output) ). For a scalar
    // loss this is simply 1.0.
    OpNode *root = order[len - 1];
    root->grad = tensor_create(tensor->shape, tensor->ndim);
    if (!root->grad) {
        free(order);
        return;
    }
    for (size_t i = 0; i < root->grad->size; i++)
        root->grad->data[i] = 1.0f;
    if (root->value)
        root->value->grad = root->grad;

    // Walk backwards: post-order lists leaves first, so the root is last.
    for (size_t i = len; i-- > 0;) {
        if (order[i]->backward)
            order[i]->backward(order[i]);
    }

    free(order);
}

/*
 * Free a single graph node (inputs/saved arrays + struct).
 * Saved/value/grad tensors are pool- or caller-owned — never freed here.
 */
void opnode_free(OpNode *node) {
    if (!node)
        return;
    free(node->inputs);
    free(node->saved);
    free(node);
}

/*
 * Free one backward graph rooted at loss.
 * Frees every intermediate node (backward != NULL) exactly once via topo
 * order and detaches value->grad_fn so pool_reset can't leave dangling
 * pointers. Leaf nodes (backward == NULL, e.g. persistent weights) are
 * kept — they are created once and reused across steps.
 * Call after sgd_step/zero_grad, before pool_reset.
 */
void opnode_free_graph(Tensor *loss) {
    if (!loss || !loss->grad_fn)
        return;

    size_t cap = 16, len = 0;
    OpNode **order = (OpNode **)malloc(cap * sizeof(OpNode *));
    if (!order)
        return;

    _topo(loss->grad_fn, &order, &len, &cap);
    for (size_t i = 0; i < len; i++)
        order[i]->visited = 0;

    for (size_t i = 0; i < len; i++) {
        OpNode *n = order[i];
        if (!n || !n->backward)
            continue; // keep leaves (persistent params)
        if (n->value)
            n->value->grad_fn = NULL;
        free(n->inputs);
        free(n->saved);
        free(n);
    }

    free(order);
}