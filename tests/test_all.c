#include "Autograd.h"
#include "memory_pool.h"
#include "ops.h"
#include "optimizer.h"
#include "tensor.h"

#include <assert.h>
#include <math.h>
#include <stdio.h>

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

int main() {
    test_tensor_create_and_fill();
    test_persistent_survives_pool_reset();
    test_sgd_step_basic();
    test_end_to_end_training_step();
    test_randn_basic();
    test_xavier_uniform_2d();
    test_xavier_normal_2d();
    printf("%d passed %d failed\n", passed, failed);
    return failed ? 1 : 0;
}