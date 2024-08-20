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

void nn_forward(NN nn)
{

}

float cost(NN nn, int data_num, float images[][IMAGE_UNIT_LEN], int labels[])
{
  return 0.0f;
}

void nn_backprop(NN nn, NN gradient, int data_num, float images[][IMAGE_UNIT_LEN], int labels[])
{
  
}

void learn(NN nn, NN gradient)
{
  
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

  size_t arch[] = {784, 100, 10};

  // initialize matrices
  Matrix input = matrix_alloc(1, 784);

  Matrix ws1 = matrix_alloc(784, 100);
  Matrix bs1 = matrix_alloc(1, 100);
  Matrix as1 = matrix_alloc(1, 100);

  Matrix ws2 = matrix_alloc(100, 10);
  Matrix bs2 = matrix_alloc(1, 10);

  Matrix output = matrix_alloc(1, 10);

  //input
  for (size_t i = 0; i < 784; i++) {
    MATRIX_AT(input, 0, i) = training_images[4][i];
  }
  printf("label: %d\n", training_labels[4]);
  
  // randomize

  matrix_rand(ws1, -1, 1);
  matrix_rand(bs1, -1, 1);

  matrix_rand(ws2, -1, 1);
  matrix_rand(bs2, -1, 1);

  // forward
  matrix_dot(as1, input, ws1);
  matrix_sum(as1, bs1);
  matrix_sig(as1);
  MATRIX_PRINT(as1);

  matrix_dot(output, as1, ws2);
  matrix_sum(output, bs2);

  MATRIX_PRINT(output);
  matrix_sig(output);
  MATRIX_PRINT(output);

  // gradient matrices
  Matrix g_input = matrix_alloc(1, 784);

  Matrix g_ws1 = matrix_alloc(784, 100);
  Matrix g_bs1 = matrix_alloc(1, 100);
  Matrix g_as1 = matrix_alloc(1, 100);

  Matrix g_ws2 = matrix_alloc(100, 10);
  Matrix g_bs2 = matrix_alloc(1, 10);

  Matrix g_output = matrix_alloc(1, 10);

  //input
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


  // get average gradient


  // update weights
  for (size_t i = 0; i < ws1.rows; i++) {
    for (size_t j = 0; j < ws1.cols; j++) {
      MATRIX_AT(ws1, i, j) -= LEARNING_RATE * MATRIX_AT(g_ws1, i, j); 
    }
  }

  for (size_t i = 0; i < ws2.rows; i++) {
    for (size_t j = 0; j < ws2.cols; j++) {
      MATRIX_AT(ws2, i, j) -= LEARNING_RATE * MATRIX_AT(g_ws2, i, j); 
    }
  }

  for (size_t i = 0; i < bs1.rows; i++) {
    for (size_t j = 0; j < bs1.cols; j++) {
      MATRIX_AT(bs1, i, j) -= LEARNING_RATE * MATRIX_AT(g_bs1, i, j); 
    }
  }

  for (size_t i = 0; i < bs2.rows; i++) {
    for (size_t j = 0; j < bs2.cols; j++) {
      MATRIX_AT(bs2, i, j) -= LEARNING_RATE * MATRIX_AT(g_bs2, i, j); 
    }
  }

  matrix_dot(as1, input, ws1);
  matrix_sum(as1, bs1);
  matrix_sig(as1);
  MATRIX_PRINT(as1);

  matrix_dot(output, as1, ws2);
  matrix_sum(output, bs2);

  MATRIX_PRINT(output);
  matrix_sig(output);
  MATRIX_PRINT(output);

  return 0;
}

