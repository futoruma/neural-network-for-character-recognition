#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define RENDER_HEIGHT 3240
#define RENDER_WIDTH 3240
uint32_t canvas_pixels[RENDER_WIDTH * RENDER_HEIGHT];

#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

#define MAX_FILEPATH_LEN 256
#define RENDER_PATH "./render/"
#define SAVED_MODELS_PATH "./saved_models/"
#define TEST_IMAGES_PATH "./test_data/test_images"
#define TEST_LABELS_PATH "./test_data/test_labels"
#define TRAINING_IMAGES_PATH "./training_data/training_images"
#define TRAINING_LABELS_PATH "./training_data/training_labels"

#define LABEL_UNIT_LEN 1
#define LABELS_METADATA_LEN 2
#define IMAGE_UNIT_LEN 784
#define IMAGES_METADATA_LEN 4
#define TEST_IMAGES_COUNT 10000
#define TRAINING_IMAGES_COUNT 60000

#define DIGITS 10
#define EPOCHS 2 
#define LEARNING_RATE 0.03f
#define TRAINING_BATCH 100

#define PNG_CHANNELS 4

float training_images[TRAINING_IMAGES_COUNT][IMAGE_UNIT_LEN];
float test_images[TEST_IMAGES_COUNT][IMAGE_UNIT_LEN];
int training_labels[TRAINING_IMAGES_COUNT];
int test_labels[TEST_IMAGES_COUNT];

void dataset_load(char *dataset_name, float images[][IMAGE_UNIT_LEN], int *labels)
{
  int images_count;
  char *images_path, *labels_path;
  if (strcmp(dataset_name, "training") == 0) {
    images_count = TRAINING_IMAGES_COUNT;
    images_path = TRAINING_IMAGES_PATH;
    labels_path = TRAINING_LABELS_PATH;
  } else if (strcmp(dataset_name, "test") == 0) {
    images_count = TEST_IMAGES_COUNT;
    images_path = TEST_IMAGES_PATH;
    labels_path = TEST_LABELS_PATH;
  } else {
    fprintf(stderr, "Unknown dataset.");
    exit(1);
  }

  int file_descriptor = open(images_path, O_RDONLY);
  if (file_descriptor == -1) {
    fprintf(stderr, "Error opening the images file.");
    exit(1);
  }

  int metadata_buf[IMAGES_METADATA_LEN]; 
  read(file_descriptor, metadata_buf, IMAGES_METADATA_LEN * sizeof(int));

  unsigned char image_buf[IMAGE_UNIT_LEN];
  for (size_t i = 0; i < images_count; ++i) {
    read(file_descriptor, image_buf, IMAGE_UNIT_LEN * sizeof(unsigned char));   
    for (size_t j = 0; j < IMAGE_UNIT_LEN; ++j)
      images[i][j]  = (float) image_buf[j] / 255.0f;
  }

  close(file_descriptor);

  file_descriptor = open(labels_path, O_RDONLY);
  if (file_descriptor == -1) {
    fprintf(stderr, "Error opening the labels file.");
    exit(1);
  }

  read(file_descriptor, metadata_buf, LABELS_METADATA_LEN * sizeof(int));

  unsigned char label_buf[LABEL_UNIT_LEN];
  for (size_t i = 0; i < images_count; ++i) {
    read(file_descriptor, label_buf, LABEL_UNIT_LEN * sizeof(unsigned char));   
    labels[i] = (int) label_buf[0];
  }

  close(file_descriptor);
}

int main(int argc, char *argv[])
{ 
  if (argc == 1) {
    fprintf(stderr, "Please specify parameters.");
    return 1;
  }

  size_t arch[] = {IMAGE_UNIT_LEN, 48, 24, DIGITS};
  size_t layer_count = ARRAY_LEN(arch);

  NN nn = nn_alloc(arch, layer_count);

  if ((argc == 2) && (strcmp(argv[1], "-train") == 0)) {
    dataset_load("training", training_images, training_labels);
    dataset_load("test", test_images, test_labels);

    NN gradient = nn_alloc(arch, layer_count);

    nn_train(nn, gradient, TRAINING_IMAGES_COUNT, IMAGE_UNIT_LEN, training_images, training_labels, EPOCHS, LEARNING_RATE, TRAINING_BATCH);

    nn_save(nn, SAVED_MODELS_PATH, LEARNING_RATE, EPOCHS);

    nn_test(nn, "training", TRAINING_IMAGES_COUNT, IMAGE_UNIT_LEN, training_images, training_labels);
    nn_test(nn, "test", TEST_IMAGES_COUNT, IMAGE_UNIT_LEN, test_images, test_labels);
  }

  else if ((argc == 3) && (strcmp(argv[1], "-test") == 0)) {
    dataset_load("training", training_images, training_labels);
    dataset_load("test", test_images, test_labels);

    nn_load(nn, SAVED_MODELS_PATH, argv[2]);
    
    nn_test(nn, "training", TRAINING_IMAGES_COUNT, IMAGE_UNIT_LEN, training_images, training_labels);
    nn_test(nn, "test", TEST_IMAGES_COUNT, IMAGE_UNIT_LEN, test_images, test_labels);
  }

  else if ((argc == 5) && (strcmp(argv[1], "-guess") == 0) && (strcmp(argv[3], "-model") == 0)) {
    printf("Guessing the number...\n");
    
    nn_load(nn, SAVED_MODELS_PATH, argv[4]);

    int file_descriptor = open(argv[2], O_RDONLY);
    if (file_descriptor == -1) {
      fprintf(stderr, "Error opening the file.");
      exit(1);
    }

    unsigned char buf[784];
  
    read(file_descriptor, buf, 13 * sizeof(unsigned char));
    read(file_descriptor, buf, 784 * sizeof(unsigned char));   
    
    close(file_descriptor);

    for (size_t i = 0; i < 784; ++i) {
      MATRIX_AT(NN_INPUT(nn), 0, i) = (float) buf[i] / 255.0f;
    }

    nn_guess(nn);

    printf("Calculated probabilities:\n");
    for (size_t i = 0; i < DIGITS; ++i) {
      printf("(%zu) -> %f %%\n", i, MATRIX_AT(NN_OUTPUT(nn), 0, i) * 100);
    }
  }

  else if ((argc == 3) && (strcmp(argv[1], "-print") == 0)) {
    char *model_name = argv[2];
    nn_load(nn, SAVED_MODELS_PATH, model_name);

    NN_PRINT(nn);
  }

  else if ((argc == 3) && (strcmp(argv[1], "-render") == 0)) {
    char *model_name = argv[2];
    nn_load(nn, SAVED_MODELS_PATH, model_name);

    Olivec_Canvas canvas = olivec_canvas(canvas_pixels, RENDER_WIDTH, RENDER_HEIGHT, RENDER_WIDTH);
    nn_render(canvas, nn);

    char canvas_filepath[MAX_FILEPATH_LEN];
    snprintf(canvas_filepath, sizeof(canvas_filepath), "%s%s.png", RENDER_PATH, model_name);

    if (!stbi_write_png(canvas_filepath, canvas.width, canvas.height, PNG_CHANNELS, canvas_pixels, canvas.stride * sizeof(uint32_t))) {
      fprintf(stderr, "Could not save the file.");
      return 1;
    }
  }

  else {
    fprintf(stderr, "Invalid parameters.");
    return 1;
  }

  return 0;
}

