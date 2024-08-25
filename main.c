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

  size_t arch[] = {IMAGE_UNIT_LEN, 48, DIGITS};
  size_t layer_count = ARRAY_LEN(arch);

  NN nn = nn_alloc(arch, layer_count);
  NN gradient = nn_alloc(arch, layer_count);

  nn_init(nn);

  for (size_t e = 0; e < EPOCHS; e++) {
    nn_zero(gradient);
    
    for (size_t d = 0; d < TRAINING_IMAGES_NUM; d++) {
      for (size_t p = 0; p < IMAGE_UNIT_LEN; p++) {
        MATRIX_AT(NN_INPUT(nn), 0, p) = training_images[d][p];
      }
      nn_forward(nn);

      for (size_t j = 0; j <= nn.count; j++) {
        matrix_fill(gradient.as[j], 0);
      }

      matrix_copy(NN_OUTPUT(gradient), NN_OUTPUT(nn));
      MATRIX_AT(NN_OUTPUT(gradient), 0, training_labels[d]) -= 1.0f;

      nn_get_total_gradient(nn, gradient);

      if ((d % 100) == 99) {
        nn_get_average_gradient(gradient, 100);
        nn_update_weights(nn, gradient, LEARNING_RATE);
        nn_zero(gradient);
      }
    }
  }

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

