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

#define TRAINING_IMAGES_NUM 60000
#define TEST_IMAGES_NUM 10000
#define IMAGE_UNIT_LEN 784
#define LABEL_UNIT_LEN 1
#define IMAGES_METADATA_LEN 4
#define LABELS_METADATA_LEN 2
#define DIGITS 10
#define EPS 0.1f
#define LEARNING_RATE 0.1f

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

void save_buffer_to_images(int data_num, float images[][IMAGE_UNIT_LEN])
{
  for (size_t i = 0; i < data_num; i++)
    for (size_t j = 0; j < IMAGE_UNIT_LEN; j++)
      images[i][j]  = (float) char_buf[i][j] / 255.0f;
}

void save_buffer_to_labels(int data_num, int labels[])
{
  for (size_t i = 0; i < data_num; i++)
    labels[i]  = (int) char_buf[i][0];
}

void nn_forward(NN nn)
{
  for (size_t i = 0; i < nn.count; i++) {
    matrix_dot(nn.as[i+1], nn.as[i], nn.ws[i]);
    matrix_sum(nn.as[i+1], nn.bs[i]);
    (i + 1) == nn.count ? matrix_softmax(nn.as[i+1]) : matrix_sig(nn.as[i+1]);
  }
}

float cost(NN nn, int data_num, float images[][IMAGE_UNIT_LEN], int labels[])
{
  float cost = 0;
  for (size_t i = 0; i < data_num; i++) {
    for (size_t j = 0; j < IMAGE_UNIT_LEN; j++) {
      MATRIX_AT(NN_INPUT(nn), 0, j) = images[i][j];
    }

    nn_forward(nn);

    int correct_digit = labels[i];

    for (size_t j = 0; j < DIGITS; j++) {
      float diff = MATRIX_AT(NN_OUTPUT(nn), 0, j);
      if (j == correct_digit) {
        diff -= 1.0f;
      }
      cost += diff * diff;
    }
  }

  return cost / data_num;
}

void finite_diff(NN nn, NN gradient, float images[][IMAGE_UNIT_LEN], int labels[])
{
  float saved;
  float c = cost(nn, 100, training_images, training_labels);

  for (size_t l = 0; l < nn.count; l++) {
    
    for (size_t i = 0; i < nn.ws[l].rows; i++) {
      for (size_t j = 0; j < nn.ws[l].cols; j++) {
        saved = MATRIX_AT(nn.ws[l], i, j);
        MATRIX_AT(nn.ws[l], i, j) += EPS;
        MATRIX_AT(gradient.ws[l], i, j) = (cost(nn, 100, training_images, training_labels) - c) / EPS; 
        MATRIX_AT(nn.ws[l], i, j) = saved;
      }
    }
    
    for (size_t i = 0; i < nn.bs[l].rows; i++) {
      for (size_t j = 0; j < nn.bs[l].cols; j++) {
        saved = MATRIX_AT(nn.bs[l], i, j);
        MATRIX_AT(nn.bs[l], i, j) += EPS;
        MATRIX_AT(gradient.bs[l], i, j) = (cost(nn, 100, training_images, training_labels) - c) / EPS; 
        MATRIX_AT(nn.bs[l], i, j) = saved;
      }
    }
  }
}

void learn(NN nn, NN gradient)
{
  for (size_t l = 0; l < nn.count; l++) {
    
    for (size_t i = 0; i < nn.ws[l].rows; i++) {
      for (size_t j = 0; j < nn.ws[l].cols; j++) {
        MATRIX_AT(nn.ws[l], i, j) -= LEARNING_RATE * MATRIX_AT(gradient.ws[l], i, j);
      }
    }
    
    for (size_t i = 0; i < nn.bs[l].rows; i++) {
      for (size_t j = 0; j < nn.bs[l].cols; j++) {
        MATRIX_AT(nn.bs[l], i, j) -= LEARNING_RATE * MATRIX_AT(gradient.bs[l], i, j);
      }
    }
  }
}

int main(void)
{  
  load_into_buffer(TRAINING_IMAGES_PATH, IMAGES_METADATA_LEN, IMAGE_UNIT_LEN, TRAINING_IMAGES_NUM);
  save_buffer_to_images(TRAINING_IMAGES_NUM, training_images);

  load_into_buffer(TEST_IMAGES_PATH, IMAGES_METADATA_LEN, IMAGE_UNIT_LEN, TEST_IMAGES_NUM);
  save_buffer_to_images(TEST_IMAGES_NUM, test_images);
    
  load_into_buffer(TRAINING_LABELS_PATH, LABELS_METADATA_LEN, LABEL_UNIT_LEN, TRAINING_IMAGES_NUM);
  save_buffer_to_labels(TRAINING_IMAGES_NUM, training_labels);
    
  load_into_buffer(TEST_LABELS_PATH, LABELS_METADATA_LEN, LABEL_UNIT_LEN, TEST_IMAGES_NUM);
  save_buffer_to_labels(TEST_IMAGES_NUM, test_labels);

  srand(time(0));

  size_t arch[] = {784, 16, 16, 10};
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN gradient = nn_alloc(arch, ARRAY_LEN(arch));

  nn_rand(nn, 0, 1);

  printf("cost = %f\n", cost(nn, 100, training_images, training_labels));
  finite_diff(nn, gradient, training_images, training_labels);
  
  learn(nn, gradient);
  printf("cost = %f\n", cost(nn, 100, training_images, training_labels));

  MATRIX_PRINT(NN_OUTPUT(nn));

  return 0;
}

