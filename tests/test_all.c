#include "Autograd.h"
#include "Module/module.h"
#include "memory_pool.h"
#include "ops.h"
#include "optimizer.h"
#include "tensor.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

static int passed = 0, failed = 0;
#define TEST(name) void test_##name()
#define ASSERT(cond)                                                                               \
    do {                                                                                           \
        if (!(cond)) {                                                                             \
            printf(" FAIL %s:%d %s\n", __FILE__, __LINE__, #cond);                                 \
            failed++;                                                                              \
            return;                                                                                \
        }                                                                                          \
    } while (0)
#define ASSERT_CLOSE(a, b, eps) ASSERT(fabs((a) - (b)) < (eps))

TEST(tensor_create_and_fill) {
    pool_reset(get_pool());
    size_t sh[2] = {2, 3};
    Tensor *t = tensor_create(sh, 2);
    ASSERT(t && t->size == 6 && t->ndim == 2);
    tensor_fill(t, 2.5f);
    for (size_t i = 0; i < 6; i++)
        ASSERT_CLOSE(t->data[i], 2.5f, 1e-6);
    passed++;
}

TEST(persistent_survives_pool_reset) {
    pool_reset(get_pool());
    size_t sh[2] = {2, 2};
    Tensor *w = tensor_persistent_create(sh, 2);
    tensor_fill(w, 3.0f);
    w->requires_grad = true;
    Tensor *tmp = tensor_create(sh, 2);
    tensor_fill(tmp, 1.0f);
    ASSERT(w->data[0] == 3.0f);
    pool_reset(get_pool());
    // tmp is invalid after reset, w must survive
    ASSERT(w->data[0] == 3.0f);
    ASSERT(w->data[3] == 3.0f);
    tensor_persistent_free(w);
    passed++;
}

TEST(sgd_step_basic) {
    pool_reset(get_pool());
    size_t sh[2] = {1, 2};
    Tensor *w = tensor_persistent_create(sh, 2);
    w->data[0] = 1.0f;
    w->data[1] = 2.0f;
    w->requires_grad = true;
    SGD *opt = sgd_create(0.1f);
    sgd_add_param(opt, w);
    // Fake grad as if backward produced [0.5, -0.5]
    size_t gsh[2] = {1, 2};
    Tensor *g = tensor_create(gsh, 2);
    g->data[0] = 0.5f;
    g->data[1] = -0.5f;
    w->grad = g;
    if (w->grad_fn)
        w->grad_fn->grad = g;
    else {
        // ensure node exists for zero_grad path
        w->grad_fn = node_of(w);
        if (w->grad_fn)
            w->grad_fn->grad = g;
    }
    sgd_step(opt);
    ASSERT_CLOSE(w->data[0], 0.95f, 1e-6); // 1 - 0.1*0.5
    ASSERT_CLOSE(w->data[1], 2.05f, 1e-6); // 2 - 0.1*(-0.5)
    sgd_zero_grad(opt);
    ASSERT(w->grad == NULL);
    sgd_free(opt);
    tensor_persistent_free(w);
    pool_reset(get_pool());
    passed++;
}

TEST(end_to_end_training_step) {
    pool_reset(get_pool());
    size_t sh_w[2] = {3, 2};
    Tensor *w = tensor_persistent_create(sh_w, 2);
    tensor_fill(w, 2.0f);
    w->requires_grad = true;
    SGD *opt = sgd_create(0.01f);
    sgd_add_param(opt, w);
    float w0_before = w->data[0];
    // One training step similar to Main.c
    Tensor *x = tensor_create((size_t[]){2, 3}, 2);
    x->data[0] = 1;
    x->data[1] = 2;
    x->data[2] = 3;
    x->data[3] = 4;
    x->data[4] = 5;
    x->data[5] = 6;
    Tensor *targets = tensor_create((size_t[]){2, 2}, 2);
    targets->data[0] = 1;
    targets->data[1] = 0;
    targets->data[2] = 0;
    targets->data[3] = 1;
    Tensor *logits = matmul(x, w);
    Tensor *probs = softmax(logits, 1);
    Tensor *loss = cross_entropy_loss(probs, targets);
    ASSERT(loss != NULL);
    float loss_before = loss->data[0];
    tensor_backward(loss);
    ASSERT(w->grad != NULL);
    sgd_step(opt);
    ASSERT(w->data[0] != w0_before); // weight moved
    sgd_zero_grad(opt);
    ASSERT(w->grad == NULL);
    pool_reset(get_pool());
    // w survives, loss computed before reset is captured
    ASSERT(loss_before > 0.3f && loss_before < 0.4f);
    sgd_free(opt);
    tensor_persistent_free(w);
    passed++;
}
TEST(randn_basic) {
    pool_reset(get_pool());
    tensor_random_seed(12345);
    size_t sh[2] = {1000, 1};
    Tensor *t = tensor_randn(sh, 2);
    ASSERT(t && t->size == 1000);
    float sum = 0.0f, sumsq = 0.0f;
    for (size_t i = 0; i < 1000; i++) {
        sum += t->data[i];
        sumsq += t->data[i] * t->data[i];
    }
    float mean = sum / 1000.0f;
    float var = sumsq / 1000.0f - mean * mean;
    ASSERT(fabsf(mean) < 0.1f);       // roughly zero
    ASSERT(fabsf(var - 1.0f) < 0.1f); // roughly unit variance
    passed++;
}

TEST(xavier_uniform_2d) {
    pool_reset(get_pool());
    tensor_random_seed(12345);
    size_t sh[2] = {64, 128};
    Tensor *t = tensor_xavier_uniform(sh, 2);
    ASSERT(t && t->size == 64 * 128);
    float max_val = 0.0f;
    for (size_t i = 0; i < t->size; i++)
        if (fabsf(t->data[i]) > max_val) max_val = fabsf(t->data[i]);
    float a = sqrtf(6.0f / ((float)(64 + 128)));
    ASSERT(max_val <= a + 1e-6f);
    passed++;
}

TEST(xavier_normal_2d) {
    pool_reset(get_pool());
    tensor_random_seed(12345);
    size_t sh[2] = {64, 128};
    Tensor *t = tensor_xavier_normal(sh, 2);
    ASSERT(t && t->size == 64 * 128);
    float sumsq = 0.0f;
    for (size_t i = 0; i < t->size; i++)
        sumsq += t->data[i] * t->data[i];
    float std = sqrtf(sumsq / (64 * 128));
    float expected_std = sqrtf(2.0f / ((float)(64 + 128)));
    ASSERT(fabsf(std - expected_std) < 0.05f);
    passed++;
}

TEST(broadcast_add_forward_bias_row) {
    pool_reset(get_pool());
    size_t shB[2] = {5, 3};
    Tensor *B = tensor_create(shB, 2);
    for (size_t i = 0; i < 15; i++) B->data[i] = (float)(int)i;
    size_t shA[1] = {3};
    Tensor *A = tensor_create(shA, 1);
    A->data[0] = 10.0f; A->data[1] = 20.0f; A->data[2] = 30.0f;

    Tensor *C = broadcast_add(A, B);
    ASSERT(C && C->ndim == 2 && C->shape[0] == 5 && C->shape[1] == 3);
    ASSERT_CLOSE(C->data[0], 10.0f + 0.0f, 1e-5f);
    ASSERT_CLOSE(C->data[1], 20.0f + 1.0f, 1e-5f);
    ASSERT_CLOSE(C->data[2], 30.0f + 2.0f, 1e-5f);
    ASSERT_CLOSE(C->data[3], 10.0f + 3.0f, 1e-5f);
    ASSERT_CLOSE(C->data[4], 20.0f + 4.0f, 1e-5f);
    pool_reset(get_pool());
    passed++;
}

TEST(linear_create_and_forward) {
    pool_reset(get_pool());
    Linear *l = linear_create(4, 3, true);
    ASSERT(l && l->W && l->b);
    ASSERT(l->W->ndim == 2 && l->W->shape[0] == 4 && l->W->shape[1] == 3);
    ASSERT(l->b->ndim == 1 && l->b->shape[0] == 3);
    ASSERT(l->W->requires_grad && l->b->requires_grad);

    size_t shx[2] = {5, 4};
    Tensor *x = tensor_create(shx, 2);
    for (size_t i = 0; i < 20; i++) x->data[i] = (float)(int)i;

    Tensor *y = l->base.forward(&l->base, x);
    ASSERT(y && y->ndim == 2 && y->shape[0] == 5 && y->shape[1] == 3);
    pool_reset(get_pool());
    linear_free(l);
    passed++;
}

TEST(linear_end_to_end_gradients) {
    pool_reset(get_pool());
    Linear *l = linear_create(3, 2, true);
    ASSERT(l && l->W && l->b);

    Tensor **params; size_t n;
    module_parameters(&l->base, &params, &n);
    ASSERT(n == 2);
    SGD *opt = sgd_create(0.1f);
    for (size_t i = 0; i < n; i++) sgd_add_param(opt, params[i]);
    free(params);

    float w00_before = l->W->data[0];
    Tensor *x = tensor_create((size_t[]){2, 3}, 2);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
    x->data[3] = 4; x->data[4] = 5; x->data[5] = 6;

    Tensor *t = tensor_create((size_t[]){2, 2}, 2);
    t->data[0] = 1; t->data[1] = 0;
    t->data[2] = 0; t->data[3] = 1;

    Tensor *y = l->base.forward(&l->base, x);
    Tensor *probs = softmax(y, 1);
    Tensor *loss = cross_entropy_loss(probs, t);
    ASSERT(loss != NULL);
    tensor_backward(loss);
    ASSERT(l->W->grad != NULL);
    ASSERT(l->b->grad != NULL);

    sgd_step(opt);
    ASSERT(l->W->data[0] != w00_before);
    sgd_zero_grad(opt);
    ASSERT(l->W->grad == NULL);
    ASSERT(l->b->grad == NULL);

    pool_reset(get_pool());
    linear_free(l);
    sgd_free(opt);
    passed++;
}

TEST(opnode_free_graph_clears_intermediates_keeps_leaf) {
    pool_reset(get_pool());
    size_t sh_w[2] = {3, 2};
    Tensor *w = tensor_persistent_create(sh_w, 2);
    tensor_fill(w, 1.0f);
    w->requires_grad = true;
    OpNode *leaf_before = node_of(w);
    ASSERT(leaf_before != NULL);

    Tensor *x = tensor_create((size_t[]){2, 3}, 2);
    x->data[0] = 1; x->data[1] = 2; x->data[2] = 3;
    x->data[3] = 4; x->data[4] = 5; x->data[5] = 6;
    Tensor *t = tensor_create((size_t[]){2, 2}, 2);
    t->data[0] = 1; t->data[1] = 0;
    t->data[2] = 0; t->data[3] = 1;

    Tensor *logits = matmul(x, w);
    Tensor *probs = softmax(logits, 1);
    Tensor *loss = cross_entropy_loss(probs, t);
    ASSERT(loss != NULL);
    tensor_backward(loss);
    ASSERT(w->grad != NULL);
    ASSERT(logits->grad_fn != NULL);

    opnode_free_graph(loss);

    // Intermediate nodes detached, persistent leaf node survives
    ASSERT(logits->grad_fn == NULL);
    ASSERT(probs->grad_fn == NULL);
    ASSERT(loss->grad_fn == NULL);
    ASSERT(w->grad_fn == leaf_before);

    // Second step still works — leaf node not dangling
    pool_reset(get_pool());
    Tensor *x2 = tensor_create((size_t[]){2, 3}, 2);
    for (size_t i = 0; i < 6; i++) x2->data[i] = (float)(i + 1);
    Tensor *t2 = tensor_create((size_t[]){2, 2}, 2);
    t2->data[0] = 1; t2->data[1] = 0;
    t2->data[2] = 0; t2->data[3] = 1;
    Tensor *logits2 = matmul(x2, w);
    Tensor *probs2 = softmax(logits2, 1);
    Tensor *loss2 = cross_entropy_loss(probs2, t2);
    ASSERT(loss2 != NULL);
    w->grad = NULL;
    if (w->grad_fn) w->grad_fn->grad = NULL;
    tensor_backward(loss2);
    ASSERT(w->grad != NULL);
    opnode_free_graph(loss2);

    pool_reset(get_pool());
    tensor_persistent_free(w);
    passed++;
}

int main() {
    test_tensor_create_and_fill();
    test_persistent_survives_pool_reset();
    test_sgd_step_basic();
    test_end_to_end_training_step();
    test_randn_basic();
    test_xavier_uniform_2d();
    test_xavier_normal_2d();
    test_broadcast_add_forward_bias_row();
    test_linear_create_and_forward();
    test_linear_end_to_end_gradients();
    test_opnode_free_graph_clears_intermediates_keeps_leaf();
    printf("%d passed %d failed\n", passed, failed);
    return failed ? 1 : 0;
}