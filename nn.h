#ifndef NN_H_
#define NN_H_

#include <assert.h>
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

typedef struct {
  size_t rows;
  size_t cols;
  float *es;
} Matrix;

typedef struct {
  size_t count;
  Matrix *ws;
  Matrix *bs;
  Matrix *as;
} NN;

float rand_float(void);
float sigmoidf(float x);

Matrix matrix_alloc(size_t rows, size_t cols);
void matrix_copy(Matrix dst, Matrix src);
void matrix_dot(Matrix dst, Matrix a, Matrix b);
void matrix_fill(Matrix matrix, float x);
void matrix_print(Matrix matrix, const char *name, size_t padding);
void matrix_rand(Matrix matrix, float low, float high);
void matrix_sig(Matrix matrix);
void matrix_softmax(Matrix matrix);
void matrix_sum(Matrix dst, Matrix a);

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_forward(NN nn);
void nn_get_average_gradient(NN gradient, size_t data_num);
void nn_get_total_gradient(NN nn, NN gradient);
void nn_init(NN nn);
void nn_print(NN nn, const char *name);
void nn_update_weights(NN nn, NN gradient, float learning_rate);
void nn_zero(NN nn);

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])
#define NN_INPUT(nn) (nn).as[0]
#define MATRIX_AT(matrix, i, j) (matrix).es[(i) * (matrix).cols + (j)] 
#define MATRIX_PRINT(m) matrix_print(m, #m, 0)
#define NN_OUTPUT(nn) (nn).as[(nn).count]
#define NN_PRINT(nn) nn_print(nn, #nn)

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float sigmoidf(float x)
{
  return 1.0f / (1.0f + expf(-x));
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
  matrix.es = malloc(sizeof(*matrix.es) * rows * cols);
  assert(matrix.es != NULL);
  return matrix;
}

void matrix_copy(Matrix dst, Matrix src)
{
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MATRIX_AT(dst, i, j) = MATRIX_AT(src, i, j);
    }
  }
}

void matrix_dot(Matrix dst, Matrix a, Matrix b)
{
  assert(a.cols == b.rows);
  size_t n = a.cols;
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MATRIX_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; k++) {
        MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j); 
      }
    }
  }
}

void matrix_fill(Matrix matrix, float x)
{
  for (size_t i = 0; i < matrix.rows; i++) {
    for (size_t j = 0; j < matrix.cols; j++) {
      MATRIX_AT(matrix, i, j) = x; 
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

void matrix_rand(Matrix matrix, float low, float high)
{
  for (size_t i = 0; i < matrix.rows; i++) {
    for (size_t j = 0; j < matrix.cols; j++) {
      MATRIX_AT(matrix, i, j) = rand_float() * (high - low) + low;
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

void matrix_sum(Matrix dst, Matrix a)
{
  assert(dst.rows == a.rows);
  assert(dst.cols == a.cols);
  for (size_t i = 0; i < dst.rows; i++) {
    for (size_t j = 0; j < dst.cols; j++) {
      MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, j);
    }
  }
}

NN nn_alloc(size_t *arch, size_t arch_count)
{
  assert(arch_count > 0);
  NN nn;
  nn.count = arch_count - 1;

  nn.ws = malloc(sizeof(*nn.ws) * nn.count);
  assert(nn.ws != NULL);
  nn.bs = malloc(sizeof(*nn.bs) * nn.count);
  assert(nn.bs != NULL);
  nn.as = malloc(sizeof(*nn.as) * (nn.count + 1));
  assert(nn.as != NULL);

  nn.as[0] = matrix_alloc(1, arch[0]); 
  for (size_t i = 1; i < arch_count; i++) {
    nn.ws[i-1] = matrix_alloc(nn.as[i-1].cols, arch[i]);
    nn.bs[i-1] = matrix_alloc(1, arch[i]);
    nn.as[i] = matrix_alloc(1, arch[i]);
  }
  return nn;
}

void nn_forward(NN nn)
{
  for (int l = 0; l < nn.count; l++) {
    matrix_dot(nn.as[l+1], nn.as[l], nn.ws[l]);
    matrix_sum(nn.as[l+1], nn.bs[l]);
    matrix_sig(nn.as[l+1]);
  }
}

void nn_get_average_gradient(NN gradient, size_t data_num)
{
  for (size_t l = 0; l < gradient.count; l++) {
    for (size_t i = 0; i < gradient.ws[l].rows; i++) {
      for (size_t j = 0; j < gradient.ws[l].cols; j++) {
        MATRIX_AT(gradient.ws[l], i, j) /= data_num;
      }
    }
    for (size_t i = 0; i < gradient.bs[l].rows; i++) {
      for (size_t j = 0; j < gradient.bs[l].cols; j++) {
        MATRIX_AT(gradient.bs[l], i, j) /= data_num;
      }
    }
  }
}

void nn_get_total_gradient(NN nn, NN gradient)
{
  for (size_t l = nn.count; l > 0; l--) {
    for (size_t j = 0; j < nn.as[l].cols; j++) {
      float a = MATRIX_AT(nn.as[l], 0, j);
      float da = MATRIX_AT(gradient.as[l], 0, j);
      MATRIX_AT(gradient.bs[l-1], 0, j) += 2 * da * a * (1 - a);
      for (size_t k = 0; k < nn.as[l-1].cols; k++) {
        float pa = MATRIX_AT(nn.as[l-1], 0, k);
        float w = MATRIX_AT(nn.ws[l-1], k, j);
        MATRIX_AT(gradient.ws[l-1], k, j) += 2 * da * a * (1 - a) * pa;
        MATRIX_AT(gradient.as[l-1], 0, k) += 2 * da * a * (1 - a) * w;
      }
    }
  }
}

void nn_init(NN nn)
{
  float limit;
  for (size_t l = 0; l < nn.count; l++) {
    limit = sqrt(6) / sqrt(nn.as[l].cols + nn.as[l+1].cols);
    matrix_rand(nn.ws[l], -limit, limit);
    matrix_fill(nn.bs[l], 0);
  }
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

void nn_update_weights(NN nn, NN gradient, float learning_rate)
{
  for (size_t l = 0; l < gradient.count; l++) {
    for (size_t i = 0; i < gradient.ws[l].rows; i++) {
      for (size_t j = 0; j < gradient.ws[l].cols; j++) {
        MATRIX_AT(nn.ws[l], i, j) -= learning_rate * MATRIX_AT(gradient.ws[l], i, j);
      }
    }
    for (size_t i = 0; i < gradient.bs[l].rows; i++) {
      for (size_t j = 0; j < gradient.bs[l].cols; j++) {
        MATRIX_AT(nn.bs[l], i, j) -= learning_rate * MATRIX_AT(gradient.bs[l], i, j);
      }
    }
  }
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

#endif // NN_IMPLEMENTATION

