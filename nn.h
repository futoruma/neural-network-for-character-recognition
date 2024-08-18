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

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])

typedef struct {
  size_t rows;
  size_t cols;
  float *es;
} Matrix;

#define MATRIX_AT(matrix, i, j) (matrix).es[(i) * (matrix).cols + (j)] 

float rand_float(void);
float sigmoidf(float x);
float cross_entropy(int expected_num, Matrix t, int K);
float *d_cross_entropy(float *y, float *t, int K);

Matrix matrix_alloc(size_t rows, size_t cols);
void matrix_fill(Matrix matrix, float x);
void matrix_rand(Matrix matrix, float low, float high);
Matrix matrix_row(Matrix matrix, size_t row);
void matrix_copy(Matrix dst, Matrix src);
void matrix_dot(Matrix dst, Matrix a, Matrix b);
void matrix_sum(Matrix dst, Matrix a);
void matrix_sig(Matrix matrix);
void matrix_softmax(Matrix matrix);
void matrix_print(Matrix matrix, const char *name, size_t padding);
#define MATRIX_PRINT(m) matrix_print(m, #m, 0)

typedef struct {
  size_t count;
  Matrix *ws;
  Matrix *bs;
  Matrix *as;
} NN;

#define NN_INPUT(nn) (nn).as[0]
#define NN_OUTPUT(nn) (nn).as[(nn).count]

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_zero(NN nn);
void nn_print(NN nn, const char *name);
#define NN_PRINT(nn) nn_print(nn, #nn)
void nn_rand(NN nn, float low, float high);
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

float cross_entropy(int expected_num, Matrix t, int K) {
  float eps = 1e-12;
  float E = 0.0f;
    
  for (size_t k = 0; k < K; k++) {
    float yk = 0.0f;
    if (k == expected_num) yk = 1.0f;
    E -= MATRIX_AT(t, 0, k) * logf(yk + eps);
    if (k == expected_num) yk = 0.0f;
  }
    
  return E;
}

float *d_cross_entropy(float *y, float *t, int K) {
  float eps = 1e-12;
  float *dE_dy = NULL;
    
  if ((dE_dy = (float *)malloc((K + 1) * sizeof(float))) == NULL) {
    exit(-1);
  }
    
  dE_dy[0] = 0.0f;
    
  for (size_t k = 1; k <= K; k++) {
    dE_dy[k] = - (t[k] / (y[k] + eps));
  }
    
  return dE_dy;
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

void matrix_print(Matrix matrix, const char *name, size_t padding)
{
  printf("%*s%s = [\n", (int) padding, "", name);
  for (size_t i = 0; i < matrix.rows; i++) {
    printf("%*s ", (int) padding, "");
    for (size_t j = 0; j < matrix.cols; j++) {
      printf("%f ", MATRIX_AT(matrix, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int) padding, "");
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

NN nn_alloc(size_t *arch, size_t arch_count)
{
  NN_ASSERT(arch_count > 0);
  NN nn;
  nn.count = arch_count - 1;

  nn.ws = NN_MALLOC(sizeof(*nn.ws) * nn.count);
  NN_ASSERT(nn.ws != NULL);
  nn.bs = NN_MALLOC(sizeof(*nn.bs) * nn.count);
  NN_ASSERT(nn.bs != NULL);
  nn.as = NN_MALLOC(sizeof(*nn.as) * (nn.count + 1));
  NN_ASSERT(nn.as != NULL);

  nn.as[0] = matrix_alloc(1, arch[0]); 
  for (size_t i = 1; i < arch_count; i++) {
    nn.ws[i-1] = matrix_alloc(nn.as[i-1].cols, arch[i]);
    nn.bs[i-1] = matrix_alloc(1, arch[i]);
    nn.as[i] = matrix_alloc(1, arch[i]);
  }

  return nn;
}

void nn_zero(NN nn)
{
  for (size_t i = 0; i < nn.count; i++) {
    matrix_fill(nn.ws[i], 0);
    matrix_fill(nn.bs[i], 0);
    matrix_fill(nn.as[i], 0);
  }
  matrix_fill(NN_OUTPUT(nn), 0);
}

void nn_print(NN nn, const char *name)
{
  char buf[256];
  printf("%s = [\n", name);
  for (size_t i = 0; i < nn.count; i++) {
    snprintf(buf, sizeof(buf), "ws%zu", i);
    matrix_print(nn.ws[i], buf, 4);
    snprintf(buf, sizeof(buf), "bs%zu", i);
    matrix_print(nn.bs[i], buf, 4);
  }
  printf("]\n");
}

void nn_rand(NN nn, float low, float high)
{
  for (size_t i = 0; i < nn.count; i++) {
    matrix_rand(nn.ws[i], low, high);
    matrix_rand(nn.bs[i], low, high);
  }
}

#endif // NN_IMPLEMENTATION

