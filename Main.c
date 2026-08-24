#include "ops.h"
#include "tensor.h"
#include "memory_pool.h"


#define N_STEPS 100


int main() {
   for (int step = 0; step < N_STEPS; step++) {

        // 1. Create this step's tensors (cheap — they come from the pool)
        Tensor* x = tensor_create((size_t[]){2, 3}, 2);
        Tensor* w = tensor_create((size_t[]){3, 2}, 2);
        tensor_fill(x, 1.0f);
        tensor_fill(w, 2.0f);

        // 2. Forward pass
        Tensor* logits = matmul(x, w);
        Tensor* probs  = softmax(logits, 1);
        // ... loss, backward, weight update, etc.

        // 3. COPY OUT anything you must keep BEFORE resetting
        //    e.g. float step_loss = loss->data[0];
        //    (gradients for the update must also be read here)

        // 4. Reclaim all pool memory for the next step
        pool_reset(get_pool());
        //    x, w, logits, probs, ... are now INVALID — do NOT use them again
    }
    return 0;
}

