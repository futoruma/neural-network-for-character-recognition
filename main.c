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
#define LEARNING_RATE 0.03f
#define EPOCHS 2 
#define TRAINING_BATCH 100

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
    labels[i] = (int) char_buf[i][0];
}
/*
void nn_train(NN nn, NN gradient, size_t image_count, size_t image_unit_len,
              float images[][image_unit_len], int labels[], size_t epochs, 
              float learning_rate, size_t training_batch)
{
  nn_init(nn);
  
  for (size_t e = 0; e < epochs; e++) {
    nn_zero(gradient);
    
    for (size_t i = 0; i < image_count; i++) {
      for (size_t j = 0; j < image_unit_len; j++) {
        MATRIX_AT(NN_INPUT(nn), 0, j) = images[i][j];
      }
      nn_forward(nn);

      for (size_t l = 0; l <= nn.count; l++) {
        matrix_fill(gradient.as[l], 0);
      }

      matrix_copy(NN_OUTPUT(gradient), NN_OUTPUT(nn));
      MATRIX_AT(NN_OUTPUT(gradient), 0, labels[i]) -= 1.0f;

      nn_get_total_gradient(nn, gradient);

      if ((i % training_batch) == (training_batch - 1)) {
        nn_get_average_gradient(gradient, training_batch);
        nn_update_weights(nn, gradient, learning_rate);
        nn_zero(gradient);
      }
    }
  }
}
*/
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

  srand(time(0));

  size_t arch[] = {IMAGE_UNIT_LEN, 48, DIGITS};
  size_t layer_count = ARRAY_LEN(arch);

  NN nn = nn_alloc(arch, layer_count);
  NN gradient = nn_alloc(arch, layer_count);

  nn_train(nn, gradient, TRAINING_IMAGES_NUM, IMAGE_UNIT_LEN, training_images,
           training_labels, EPOCHS, LEARNING_RATE, TRAINING_BATCH);

  // check 
  size_t correct_count = 0;
  size_t max_digit = 0;
  float max_value = 0.0f;

  for (size_t d = 0; d < TRAINING_IMAGES_NUM; d++) {
    for (size_t p = 0; p < IMAGE_UNIT_LEN; p++) {
      MATRIX_AT(NN_INPUT(nn), 0, p) = training_images[d][p];
    }
    nn_forward(nn);

    max_digit = 0;
    max_value = 0.0f;

    for (size_t i = 0; i < NN_OUTPUT(nn).cols; i++) {
      if (MATRIX_AT(NN_OUTPUT(nn), 0, i) > max_value) {
        max_value = MATRIX_AT(NN_OUTPUT(nn), 0, i);
        max_digit = i;
      }
    }

    if (max_digit == training_labels[d]) {
      correct_count++;
    } 
  }

  printf("training set > correct: %zu / %d\n", correct_count, TRAINING_IMAGES_NUM);

  correct_count = 0;
  max_digit = 0;
  max_value = 0.0f;

  for (size_t d = 0; d < TEST_IMAGES_NUM; d++) {
    for (size_t p = 0; p < IMAGE_UNIT_LEN; p++) {
      MATRIX_AT(NN_INPUT(nn), 0, p) = test_images[d][p];
    }
    nn_forward(nn);

    max_digit = 0;
    max_value = 0.0f;

    for (size_t i = 0; i < NN_OUTPUT(nn).cols; i++) {
      if (MATRIX_AT(NN_OUTPUT(nn), 0, i) > max_value) {
        max_value = MATRIX_AT(NN_OUTPUT(nn), 0, i);
        max_digit = i;
      }
    }

    if (max_digit == test_labels[d]) {
      correct_count++;
    } 
  }

  printf("test set > correct: %zu / %d\n", correct_count, TEST_IMAGES_NUM);

  return 0;
}

