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

typedef struct {
  Matrix a0;
  Matrix w1, b1, a1;
  Matrix w2, b2, a2;
  Matrix w3, b3, a3;
} Model;

void forward(Model model)
{
  matrix_dot(model.a1, model.a0, model.w1);
  matrix_sum(model.a1, model.b1);
  matrix_sig(model.a1);
  
  matrix_dot(model.a2, model.a1, model.w2);
  matrix_sum(model.a2, model.b2);
  matrix_sig(model.a2);

  matrix_dot(model.a3, model.a2, model.w3);
  matrix_sum(model.a3, model.b3);
  matrix_softmax(model.a3);
}

float cost(Model model, int data_num, float images[][IMAGE_UNIT_LEN], int labels[])
{
  float cost = 0;
  for (size_t i = 0; i < data_num; i++) {
    
    for (size_t j = 0; j < IMAGE_UNIT_LEN; j++) {
      MATRIX_AT(model.a0, 0, j) = images[i][j];
    }

    forward(model);

    int correct_digit = labels[i];
    MATRIX_AT(model.a3, 0, correct_digit) -= 1.0f;

    for (size_t j = 0; j < DIGITS; j++) {
      float diff = MATRIX_AT(model.a3, 0, j);
      cost += diff * diff;
    }
  }

  return cost / data_num;
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

  Model model;

  model.a0 = matrix_alloc(1, 784);

  model.w1 = matrix_alloc(784, 18);
  model.b1 = matrix_alloc(1, 18);
  model.a1 = matrix_alloc(1, 18);

  model.w2 = matrix_alloc(18, 18);
  model.b2 = matrix_alloc(1, 18);
  model.a2 = matrix_alloc(1, 18);

  model.w3 = matrix_alloc(18, 10);
  model.b3 = matrix_alloc(1, 10);
  model.a3 = matrix_alloc(1, 10);

  matrix_rand(model.w1, 0, 1);
  matrix_rand(model.b1, 0, 1);
  matrix_rand(model.w2, 0, 1);
  matrix_rand(model.b2, 0, 1);
  matrix_rand(model.w3, 0, 1);
  matrix_rand(model.b3, 0, 1);

  printf("cost = %f\n", cost(model, 100, training_images, training_labels));
 
  MATRIX_PRINT(model.a3);

  return 0;
}

