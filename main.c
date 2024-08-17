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

int images_metadata[IMAGES_METADATA_LEN];
int labels_metadata[LABELS_METADATA_LEN];

unsigned char training_images_buffer[TRAINING_IMAGES_NUM][IMAGE_UNIT_LEN];
unsigned char test_images_buffer[TEST_IMAGES_NUM][IMAGE_UNIT_LEN];
unsigned char training_labels_buffer[TRAINING_IMAGES_NUM][LABEL_UNIT_LEN];
unsigned char test_labels_buffer[TEST_IMAGES_NUM][LABEL_UNIT_LEN];

float training_images[TRAINING_IMAGES_NUM][IMAGE_UNIT_LEN];
float test_images[TEST_IMAGES_NUM][IMAGE_UNIT_LEN];
int training_labels[TRAINING_IMAGES_NUM];
int test_labels[TEST_IMAGES_NUM];

void load_into_buffer(char *file_path, int metadata_len, int metadata[], int unit_len, int data_num, unsigned char buffer[][unit_len])
{
  int file_descriptor = open(file_path, O_RDONLY);
  if (file_descriptor == -1) {
    fprintf(stderr, "Error opening the file.");
    exit(-1);
  }
  
  read(file_descriptor, metadata, metadata_len * sizeof(int));
    
  for (size_t i = 0; i < data_num; i++) {
    read(file_descriptor, buffer[i], unit_len * sizeof(unsigned char));   
  }

  close(file_descriptor);
}

void save_buffer_to_images(int data_num, unsigned char buffer[][IMAGE_UNIT_LEN], float images[][IMAGE_UNIT_LEN])
{
  for (size_t i = 0; i < data_num; i++)
    for (size_t j = 0; j < IMAGE_UNIT_LEN; j++)
      images[i][j]  = (float) buffer[i][j] / 255.0f;
}

void save_buffer_to_labels(int data_num, unsigned char buffer[][LABEL_UNIT_LEN], int labels[])
{
  for (size_t i = 0; i < data_num; i++)
    labels[i]  = (int) buffer[i][0];
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
    MATRIX_AT(NN_OUTPUT(nn), 0, correct_digit) -= 1.0f;

    for (size_t j = 0; j < DIGITS; j++) {
      float diff = MATRIX_AT(NN_OUTPUT(nn), 0, j);
      cost += diff * diff;
    }
  }

  return cost / data_num;
}

void finite_diff(NN nn, NN gradient, float eps, float images[][IMAGE_UNIT_LEN], int labels[])
{
  float saved;
  float c = cost(nn, 100, training_images, training_labels);

  for (size_t l = 0; l < nn.count; l++) {
    
    for (size_t i = 0; i < nn.ws[l].rows; i++) {
      for (size_t j = 0; j < nn.ws[l].cols; j++) {
        saved = MATRIX_AT(nn.ws[l], i, j);
        MATRIX_AT(nn.ws[l], i, j) += eps;
        MATRIX_AT(gradient.ws[l], i, j) = (cost(nn, 100, training_images, training_labels) - c) / eps; 
        MATRIX_AT(nn.ws[l], i, j) = saved;
      }
    }
    
    for (size_t i = 0; i < nn.bs[l].rows; i++) {
      for (size_t j = 0; j < nn.bs[l].cols; j++) {
        saved = MATRIX_AT(nn.bs[l], i, j);
        MATRIX_AT(nn.bs[l], i, j) += eps;
        MATRIX_AT(gradient.bs[l], i, j) = (cost(nn, 100, training_images, training_labels) - c) / eps; 
        MATRIX_AT(nn.bs[l], i, j) = saved;
      }
    }
  }
}

void learn(NN nn, NN gradient, float rate)
{
  for (size_t l = 0; l < nn.count; l++) {
    
    for (size_t i = 0; i < nn.ws[l].rows; i++) {
      for (size_t j = 0; j < nn.ws[l].cols; j++) {
        MATRIX_AT(nn.ws[l], i, j) -= rate * MATRIX_AT(gradient.ws[l], i, j);
      }
    }
    
    for (size_t i = 0; i < nn.bs[l].rows; i++) {
      for (size_t j = 0; j < nn.bs[l].cols; j++) {
        MATRIX_AT(nn.bs[l], i, j) -= rate * MATRIX_AT(gradient.bs[l], i, j);
      }
    }
  }
}

int main(void)
{  
  load_into_buffer(TRAINING_IMAGES_PATH, IMAGES_METADATA_LEN, images_metadata, IMAGE_UNIT_LEN, TRAINING_IMAGES_NUM, training_images_buffer);
  save_buffer_to_images(TRAINING_IMAGES_NUM, training_images_buffer, training_images);

  load_into_buffer(TEST_IMAGES_PATH, IMAGES_METADATA_LEN, images_metadata, IMAGE_UNIT_LEN, TEST_IMAGES_NUM, test_images_buffer);
  save_buffer_to_images(TEST_IMAGES_NUM, test_images_buffer, test_images);
    
  load_into_buffer(TRAINING_LABELS_PATH, LABELS_METADATA_LEN, labels_metadata, LABEL_UNIT_LEN, TRAINING_IMAGES_NUM, training_labels_buffer);
  save_buffer_to_labels(TRAINING_IMAGES_NUM, training_labels_buffer, training_labels);
    
  load_into_buffer(TEST_LABELS_PATH, LABELS_METADATA_LEN, labels_metadata, LABEL_UNIT_LEN, TEST_IMAGES_NUM, test_labels_buffer);
  save_buffer_to_labels(TEST_IMAGES_NUM, test_labels_buffer, test_labels);

  srand(time(0));

  size_t arch[] = {784, 16, 16, 10};
  NN nn = nn_alloc(arch, ARRAY_LEN(arch));
  NN gradient = nn_alloc(arch, ARRAY_LEN(arch));

  nn_rand(nn, 0, 1);

  float eps = 0.1f;
  float rate = 0.1f;

  printf("cost = %f\n", cost(nn, 100, training_images, training_labels));
  finite_diff(nn, gradient, eps, training_images, training_labels);
  
  learn(nn, gradient, rate);
  printf("cost = %f\n", cost(nn, 100, training_images, training_labels));

  MATRIX_PRINT(NN_OUTPUT(nn));

  return 0;
}

