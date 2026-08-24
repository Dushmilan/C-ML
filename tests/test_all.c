#include "memory_pool.h"
#include "ops.h"
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
int main() {
    test_tensor_create_and_fill();
    printf("%d passed %d failed\n", passed, failed);
    return failed ? 1 : 0;
}