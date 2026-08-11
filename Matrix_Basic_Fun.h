#ifndef MATRIX_BASIC_FUN_H
#define MATRIX_BASIC_FUN_H

#include <stdio.h>

void matmul(int r1, int c1, int mat1[r1][c1],
            int r2, int c2, int mat2[r2][c2],
            int result[r1][c2]);

void displayMatrix(int rows, int cols, int matrix[rows][cols]);

#endif
