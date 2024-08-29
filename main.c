#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define NN_IMPLEMENTATION
#include "nn.h"

#define SAVED_MODELS_PATH "./saved_models/"
#define TEST_IMAGES_PATH "./test_data/test_images"
#define TEST_LABELS_PATH "./test_data/test_labels"
#define TRAINING_IMAGES_PATH "./training_data/training_images"
#define TRAINING_LABELS_PATH "./training_data/training_labels"

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

void nn_save(NN nn, char *save_path)
{
  char fullname[128];
  FILE *fptr;

  time_t current_time = time(NULL);
  char date_string[20];
  strftime(date_string, 20, "%Y%m%d%H%M%S", localtime(&current_time));

  strcpy(fullname, save_path);
  strcat(fullname, date_string);
  strcat(fullname, "_784x48x10");

  fptr = fopen(fullname, "wb");
  if ((fptr = fopen(fullname, "wb")) == NULL) {
    fprintf(stderr, "Error opening the file.");
    exit(1);
  }

  float buf[784];
  for (size_t l = 0; l < nn.count; l++) {
    for (size_t i = 0; i < nn.ws[l].rows; i++) {
      for (size_t j = 0; j < nn.ws[l].cols; j++) {
        buf[j] = MATRIX_AT(nn.ws[l], i, j);
      }
      fwrite(buf, sizeof(float), nn.ws[l].cols, fptr);
    }
    for (size_t i = 0; i < nn.bs[l].rows; i++) {
      for (size_t j = 0; j < nn.bs[l].cols; j++) {
        buf[j] = MATRIX_AT(nn.bs[l], i, j);
      }
      fwrite(buf, sizeof(float), nn.bs[l].cols, fptr);
    }
  }

  fclose(fptr);
}

void nn_load(NN nn, char* save_path, char *filename)
{
  char fullname[256];
  strcpy(fullname, save_path);
  strcat(fullname, filename);

  int file_descriptor = open(fullname, O_RDONLY);
  if (file_descriptor == -1) {
    fprintf(stderr, "Error loading the model.");
    exit(-1);
  }

  float buf[784];

  for (size_t l = 0; l < nn.count; l++) {
    for (size_t i = 0; i < nn.ws[l].rows; i++) {
      read(file_descriptor, buf, nn.ws[l].cols * sizeof(float));
      for (size_t j = 0; j < nn.ws[l].cols; j++) {
        MATRIX_AT(nn.ws[l], i, j) = buf[j];
      }
    }
    for (size_t i = 0; i < nn.bs[l].rows; i++) {
      read(file_descriptor, buf, nn.bs[l].cols * sizeof(float));
      for (size_t j = 0; j < nn.bs[l].cols; j++) {
        MATRIX_AT(nn.bs[l], i, j) = buf[j];
      }
    }
  }

  close(file_descriptor);
}

int main(int argc, char* argv[])
{ 
  if (argc == 1) {
    printf("Please specify parameters.\n");
    return 0;
  }

  size_t arch[] = {IMAGE_UNIT_LEN, 48, DIGITS};
  size_t layer_count = ARRAY_LEN(arch);

  NN nn = nn_alloc(arch, layer_count);

  if ((argc == 2) && (strcmp(argv[1], "-train") == 0)) {
    printf("Training model...\n");
    
    load_into_buffer(TRAINING_IMAGES_PATH, IMAGES_METADATA_LEN, 
                     IMAGE_UNIT_LEN, TRAINING_IMAGES_NUM);
    populate_images(TRAINING_IMAGES_NUM, training_images);
    load_into_buffer(TRAINING_LABELS_PATH, LABELS_METADATA_LEN, 
                     LABEL_UNIT_LEN, TRAINING_IMAGES_NUM);
    populate_labels(TRAINING_IMAGES_NUM, training_labels);
    
    NN gradient = nn_alloc(arch, layer_count);

    srand(time(0));

    nn_train(nn, gradient, TRAINING_IMAGES_NUM, IMAGE_UNIT_LEN, training_images,
             training_labels, EPOCHS, LEARNING_RATE, TRAINING_BATCH);

    nn_save(nn, SAVED_MODELS_PATH);

    printf("Model has been trained and saved.\n");

    nn_test(nn, "training", TRAINING_IMAGES_NUM, IMAGE_UNIT_LEN, 
            training_images, training_labels);
  }

  else if ((argc == 3) && (strcmp(argv[1], "-test") == 0)) {
    printf("Testing saved model...\n");
    
    load_into_buffer(TEST_IMAGES_PATH, IMAGES_METADATA_LEN, 
                     IMAGE_UNIT_LEN, TEST_IMAGES_NUM);
    populate_images(TEST_IMAGES_NUM, test_images);
      
    load_into_buffer(TEST_LABELS_PATH, LABELS_METADATA_LEN, 
                     LABEL_UNIT_LEN, TEST_IMAGES_NUM);
    populate_labels(TEST_IMAGES_NUM, test_labels);

    nn_load(nn, SAVED_MODELS_PATH, argv[2]);
    
    nn_test(nn, "test", TEST_IMAGES_NUM, IMAGE_UNIT_LEN, 
            test_images, test_labels);
  }

  else if ((argc == 3) && (strcmp(argv[1], "-print") == 0)) {
    printf("Printing saved model...\n");

    nn_load(nn, SAVED_MODELS_PATH, argv[2]);

    NN_PRINT(nn);
  }

  else {
    printf("Invalid parameters.\n");
  }

  return 0;
}

