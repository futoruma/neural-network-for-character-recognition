#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <unistd.h>

#define NN_IMPLEMENTATION
#include "nn.h"

#define TRAINING_IMAGES_PATH "./training_data/training_images"
#define TRAINING_LABELS_PATH "./training_data/training_labels"
#define TEST_IMAGES_PATH "./test_data/test_images"
#define TEST_LABELS_PATH "./test_data/test_labels"

#define TRAINING_IMAGES_NUM 5
#define TEST_IMAGES_NUM 5
#define IMAGE_UNIT_LEN 784
#define LABEL_UNIT_LEN 1
#define IMAGES_METADATA_LEN 4
#define LABELS_METADATA_LEN 2
#define DIGITS 10
#define LEARNING_RATE 0.01f
#define EPOCHS 1 

int int_buf[IMAGES_METADATA_LEN];
unsigned char char_buf[TRAINING_IMAGES_NUM][IMAGE_UNIT_LEN];

float training_images[TRAINING_IMAGES_NUM][IMAGE_UNIT_LEN];
float test_images[TEST_IMAGES_NUM][IMAGE_UNIT_LEN];
int training_labels[TRAINING_IMAGES_NUM];
int test_labels[TEST_IMAGES_NUM];

void load_into_buffer(char *file_path, int meta_len, int unit_len, int data_num)
{
  int file_descriptor = open(file_path, O_RDONLY);
  if (file_descriptor == -1) {
    fprintf(stderr, "Error opening the file.");
    exit(-1);
  }
  
  read(file_descriptor, int_buf, meta_len * sizeof(int));
    
  for (size_t i = 0; i < data_num; i++) {
    read(file_descriptor, char_buf[i], unit_len * sizeof(unsigned char));   
  }

  close(file_descriptor);
}

void populate_images(int data_num, float images[][IMAGE_UNIT_LEN])
{
  for (size_t i = 0; i < data_num; i++)
    for (size_t j = 0; j < IMAGE_UNIT_LEN; j++)
      images[i][j]  = (float) char_buf[i][j] / 255.0f;
}

void populate_labels(int data_num, int labels[])
{
  for (size_t i = 0; i < data_num; i++)
    labels[i]  = (int) char_buf[i][0];
}

float cost(NN nn, int data_num, float images[][IMAGE_UNIT_LEN], int labels[])
{
  return 0.0f;
}

void nn_get_gradient(NN nn, NN gradient)
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

void nn_average_gradient(NN gradient, size_t data_num)
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

int main(void)
{  
  load_into_buffer(TRAINING_IMAGES_PATH, IMAGES_METADATA_LEN, IMAGE_UNIT_LEN, TRAINING_IMAGES_NUM);
  populate_images(TRAINING_IMAGES_NUM, training_images);

  load_into_buffer(TEST_IMAGES_PATH, IMAGES_METADATA_LEN, IMAGE_UNIT_LEN, TEST_IMAGES_NUM);
  populate_images(TEST_IMAGES_NUM, test_images);
    
  load_into_buffer(TRAINING_LABELS_PATH, LABELS_METADATA_LEN, LABEL_UNIT_LEN, TRAINING_IMAGES_NUM);
  populate_labels(TRAINING_IMAGES_NUM, training_labels);
    
  load_into_buffer(TEST_LABELS_PATH, LABELS_METADATA_LEN, LABEL_UNIT_LEN, TEST_IMAGES_NUM);
  populate_labels(TEST_IMAGES_NUM, test_labels);

  /*
  for (size_t i = 0; i < TRAINING_IMAGES_NUM; i++) {
    printf("training label %d\n", training_labels[i]);
    for (size_t j = 1; j < 785; j++) {
      printf("%1.1f ", training_images[i][j]);
      if (j % 28 == 0) printf("\n");
    }
    printf("\n");
  }

  for (size_t i = 0; i < TEST_IMAGES_NUM; i++) {
    printf("test label %d\n", test_labels[i]);
    for (size_t j = 1; j < 785; j++) {
      printf("%1.1f ", test_images[i][j]);
      if (j % 28 == 0) printf("\n");
    }
    printf("\n");
  }
  */

  srand(time(0));

  size_t arch[] = {IMAGE_UNIT_LEN, 100, DIGITS};
  size_t layer_count = ARRAY_LEN(arch);

  NN nn = nn_alloc(arch, layer_count);
  NN gradient = nn_alloc(arch, layer_count);

  for (size_t i = 0; i < IMAGE_UNIT_LEN; i++) {
    MATRIX_AT(NN_INPUT(nn), 0, i) = training_images[0][i];
  }
  printf("label: %d\n", training_labels[0]);

  nn_rand(nn, -1, 1);

  nn_forward(nn);
  MATRIX_PRINT(nn.as[2]);

  for (size_t e = 0; e < 100; e++) {
    for (size_t j = 0; j <= nn.count; j++) {
      matrix_fill(gradient.as[j], 0);
    }

    matrix_copy(NN_OUTPUT(gradient), NN_OUTPUT(nn));
    MATRIX_AT(NN_OUTPUT(gradient), 0, training_labels[0]) -= 1.0f;

    nn_get_gradient(nn, gradient);

    nn_average_gradient(gradient, 1);

    nn_update_weights(nn, gradient, LEARNING_RATE);

    nn_zero(gradient);

    nn_forward(nn);
    MATRIX_PRINT(nn.as[2]);
  }

 
/*
  // backprop
  MATRIX_AT(g_output, 0, training_labels[4]) -= 1.0f;

  for (size_t j = 0; j < 10; j++) {
    float a = MATRIX_AT(output, 0, j);
    float da = MATRIX_AT(g_output, 0, j);
    MATRIX_AT(g_bs2, 0, j) += 2 * da * a * (1 - a);
    for (size_t k = 0; k < 100; k++) {
      float pa = MATRIX_AT(as1, 0, k);
      float w = MATRIX_AT(ws2, k, j);
      MATRIX_AT(g_ws2, k, j) += 2 * da * a * (1 - a) * pa;
      MATRIX_AT(g_as1, 0, k) += 2 * da * a * (1 - a) * w;
    }
  }
  
  for (size_t j = 0; j < 100; j++) {
    float a = MATRIX_AT(as1, 0, j);
    float da = MATRIX_AT(g_as1, 0, j);
    MATRIX_AT(g_bs1, 0, j) += 2 * da * a * (1 - a);
    for (size_t k = 0; k < 784; k++) {
      float pa = MATRIX_AT(g_input, 0, k);
      float w = MATRIX_AT(ws1, k, j);
      MATRIX_AT(g_ws1, k, j) += 2 * da * a * (1 - a) * pa;
      MATRIX_AT(g_input, 0, k) += 2 * da * a * (1 - a) * w;
    }
  }

*/
  return 0;
}

