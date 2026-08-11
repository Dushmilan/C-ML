#include "Matrix_Basic_Fun.h"

int main() {
    int mat1[2][3] = {{1, 2, 3}, {4, 5, 6}};
    int mat2[3][2] = {{7, 8}, {9, 10}, {11, 12}};
    int result[2][2];

    matmul(2, 3, mat1, 3, 2, mat2, result);

    displayMatrix(2, 2, result);

    return 0;
}

/*
TODOs:
- [x] tensor.c: implement tensor_create, tensor_free, tensor_fill, tensor_print, tensor_clone
- [x] ops.c: implement matmul, add, relu, softmax, cross_entropy_loss
- [ ] autograd.c: implement OpNode creation and backward graph traversal
- [ ] memory_pool.c: implement pool_create, pool_alloc, pool_reset
- [ ] Integrate memory pool into tensor.c (replace malloc/calloc)
- [ ] Refactor Main.c to use Tensor API instead of int mat1[2][3]
- [ ] Add tests for tensor operations
*/