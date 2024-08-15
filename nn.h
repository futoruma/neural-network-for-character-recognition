#ifndef NN_H_
#define NN_H_

#include <stddef.h>
#include <stdio.h>
#include <math.h>

#ifndef NN_MALLOC
#include <stdlib.h>
#define NN_MALLOC malloc
#endif // NN_MALLOC

#ifndef NN_ASSERT
#include <assert.h>
#define NN_ASSERT assert
#endif // NN_ASSERT

typedef struct {
  size_t rows;
  size_t cols;
  float *es;
} Matrix;

#define MATRIX_AT(matrix, i, j) (matrix).es[(i) * (matrix).cols + (j)] 

float rand_float(void);
float sigmoidf(float x);

Matrix matrix_alloc(size_t rows, size_t cols);
void matrix_fill(Matrix matrix, float x);
void matrix_rand(Matrix matrix, float low, float high);
Matrix matrix_row(Matrix matrix, size_t row);
void matrix_copy(Matrix dst, Matrix src);
void matrix_dot(Matrix dst, Matrix a, Matrix b);
void matrix_sum(Matrix dst, Matrix a);
void matrix_sig(Matrix matrix);
void matrix_softmax(Matrix matrix);
void matrix_print(Matrix matrix, const char *name);
#define MATRIX_PRINT(m) matrix_print(m, #m)

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float sigmoidf(float x)
{
  return 1.f / (1.f + expf(-x));
}

float rand_float(void)
{
  return (float) rand() / (float) RAND_MAX;
}
Matrix matrix_alloc(size_t rows, size_t cols)
{
  Matrix matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.es = NN_MALLOC(sizeof(*matrix.es) * rows * cols);
  NN_ASSERT(matrix.es != NULL);
  return matrix;
}

void matrix_dot(Matrix dst, Matrix a, Matrix b)
{
  NN_ASSERT(a.cols == b.rows);
  size_t n = a.cols;
  NN_ASSERT(dst.rows == a.rows);
  NN_ASSERT(dst.cols == b.cols);
  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MATRIX_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; k++) {
        MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j); 
      }
    }
  }
}

Matrix matrix_row(Matrix matrix, size_t row)
{
  return (Matrix){
    .rows = 1,
    .cols = matrix.cols,
    .es = &MATRIX_AT(matrix, row, 0)
  };
}

void matrix_copy(Matrix dst, Matrix src)
{
  NN_ASSERT(dst.rows == src.rows);
  NN_ASSERT(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MATRIX_AT(dst, i, j) = MATRIX_AT(src, i, j);
    }
  }
}

void matrix_sum(Matrix dst, Matrix a)
{
  NN_ASSERT(dst.rows == a.rows);
  NN_ASSERT(dst.cols == a.cols);
  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, j);
    }
  }
}

void matrix_sig(Matrix matrix)
{
  for (size_t i = 0; i < matrix.rows; i++) {
    for (size_t j = 0; j < matrix.cols; j++) {
      MATRIX_AT(matrix, i, j) = sigmoidf(MATRIX_AT(matrix, i, j));
    }
  }
}

void matrix_softmax(Matrix matrix)
{
  float max_value = 0;
  for (size_t i = 0; i < matrix.rows; i++) {
    for (size_t j = 0; j < matrix.cols; j++) {
      if (MATRIX_AT(matrix, i, j) > max_value) {
        max_value = MATRIX_AT(matrix, i, j);
      }
    }
  }
    
  float exp_sum = 0;
  for (size_t i = 0; i < matrix.rows; i++) {
    for (size_t j = 0; j < matrix.cols; j++) {
      MATRIX_AT(matrix, i, j) = expf(MATRIX_AT(matrix, i, j) - max_value);
      exp_sum += MATRIX_AT(matrix, i, j);
    }
  }

  for (size_t i = 0; i < matrix.rows; i++) {
    for (size_t j = 0; j < matrix.cols; j++) {
      MATRIX_AT(matrix, i, j) /= exp_sum;
    }
  }
}

void matrix_print(Matrix matrix, const char *name)
{
  printf("%s = [\n", name);
  for (size_t i = 0; i < matrix.rows; i++) {
    for (size_t j = 0; j < matrix.cols; j++) {
      printf("%f ", MATRIX_AT(matrix, i, j));
    }
    printf("\n");
  }
  printf("]\n");
}

void matrix_fill(Matrix matrix, float x)
{
  for (size_t i = 0; i < matrix.rows; i++) {
    for (size_t j = 0; j < matrix.cols; j++) {
      MATRIX_AT(matrix, i, j) = x; 
    }
  }
}

void matrix_rand(Matrix matrix, float low, float high)
{
  for (size_t i = 0; i < matrix.rows; i++) {
    for (size_t j = 0; j < matrix.cols; j++) {
      MATRIX_AT(matrix, i, j) = rand_float() * (high - low) + low;
    }
  }
}

#endif // NN_IMPLEMENTATION

