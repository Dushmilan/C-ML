#include "Autograd.h"
#include "memory_pool.h"
#include "ops.h"
#include "optimizer.h"
#include "tensor.h"

#include <stdio.h>

#define N_STEPS 5

int main() {
    tensor_random_seed(42);
    // Persistent weight — survives pool_reset, updated by SGD
    Tensor *w_init = tensor_xavier_uniform((size_t[]){3, 2}, 2);
    Tensor *w = tensor_persistent_create((size_t[]){3, 2}, 2);
    memcpy(w->data, w_init->data, 3 * 2 * sizeof(float));
    w->requires_grad = true;

    SGD *opt = sgd_create(0.01f);
    sgd_add_param(opt, w);

    printf("Initial w: ");
    tensor_print(w);

    for (int step = 0; step < N_STEPS; step++) {
        // Activations — pool-allocated, cheap, reclaimed each step
        Tensor *x = tensor_create((size_t[]){2, 3}, 2);
        // Diverse inputs so grad doesn't cancel (previous uniform x gave zero w grad)
        x->data[0] = 1.0f;
        x->data[1] = 2.0f;
        x->data[2] = 3.0f;
        x->data[3] = 4.0f;
        x->data[4] = 5.0f;
        x->data[5] = 6.0f;

        // One-hot targets for the 2 samples (2 classes)
        Tensor *targets = tensor_create((size_t[]){2, 2}, 2);
        targets->data[0] = 1.0f;
        targets->data[1] = 0.0f;
        targets->data[2] = 0.0f;
        targets->data[3] = 1.0f;

        // Forward: logits -> probs -> loss
        Tensor *logits = matmul(x, w);
        Tensor *probs = softmax(logits, 1);
        Tensor *loss = cross_entropy_loss(probs, targets);

        float loss_val = loss ? loss->data[0] : -1.0f;

        // Backward: populates w->grad (pool-allocated)
        if (loss)
            tensor_backward(loss);

        float grad0_before = w->grad ? w->grad->data[0] : 0.0f;
        float w_before = w->data[0];

        // Optimizer reads pool grad before it is reclaimed
        sgd_step(opt);

        float w_after = w->data[0];
        sgd_zero_grad(opt);

        printf("step %d: loss=%.6f  w[0] %.6f -> %.6f (grad %.6f) %s\n", step, loss_val, w_before,
               w_after, grad0_before, w->grad ? "grad still present" : "cleared");

        pool_reset(get_pool());
        // x, logits, probs, targets, loss now INVALID — w persists
    }

    printf("Final w: ");
    tensor_print(w);

    tensor_persistent_free(w);
    sgd_free(opt);
    test_randn_basic();
    test_xavier_uniform_2d();
    return 0;
}
