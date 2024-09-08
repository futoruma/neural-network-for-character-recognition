#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define DIGITS 10
#define EPOCHS 2 
#define HIDDEN_LAYERS 48, 24
#define LEARNING_RATE 0.03f
#define TRAINING_BATCH 100

#define LABEL_UNIT_LEN 1
#define LABELS_METADATA_LEN 2
#define IMAGE_UNIT_LEN 784
#define IMAGES_METADATA_LEN 4
#define TEST_IMAGES_COUNT 10000
#define TRAINING_IMAGES_COUNT 60000

#define MAX_FILEPATH_LEN 256
#define RENDER_PATH "./render/"
#define SAVED_MODELS_PATH "./saved_models/"
#define TEST_IMAGES_PATH "./test_data/test_images"
#define TEST_LABELS_PATH "./test_data/test_labels"
#define TRAINING_IMAGES_PATH "./training_data/training_images"
#define TRAINING_LABELS_PATH "./training_data/training_labels"

#define MAX_BRIGHTNESS_F 255.0f
#define PGM_P5_METADATA_LEN 13
#define PNG_CHANNELS 4
#define RENDER_HEIGHT 3240
#define RENDER_WIDTH 3240
uint32_t canvas_pixels[RENDER_WIDTH * RENDER_HEIGHT];

#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define NN_IMPLEMENTATION
#include "nn.h"

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
      images[i][j]  = (float) image_buf[j] / MAX_BRIGHTNESS_F;
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

void pmg_load(char *image_path, NN nn)
{
  int file_descriptor = open(image_path, O_RDONLY);
  if (file_descriptor == -1) {
    fprintf(stderr, "Error opening the file.");
    exit(1);
  }

  unsigned char buf[IMAGE_UNIT_LEN];

  read(file_descriptor, buf, PGM_P5_METADATA_LEN * sizeof(unsigned char));
  read(file_descriptor, buf, IMAGE_UNIT_LEN * sizeof(unsigned char));   
  
  close(file_descriptor);

  for (size_t i = 0; i < IMAGE_UNIT_LEN; ++i) {
    MATRIX_AT(NN_INPUT(nn), 0, i) = (float) buf[i] / MAX_BRIGHTNESS_F;
  }
}

int main(int argc, char *argv[])
{ 
  size_t arch[] = {IMAGE_UNIT_LEN, HIDDEN_LAYERS, DIGITS};
  size_t layer_count = ARRAY_LEN(arch);

  NN nn = nn_alloc(arch, layer_count);

  if ((argc == 2) && (strcmp(argv[1], "-train") == 0)) {
    NN gradient = nn_alloc(arch, layer_count);

    dataset_load("training", training_images, training_labels);
    nn_train(nn, gradient, TRAINING_IMAGES_COUNT, IMAGE_UNIT_LEN, training_images, training_labels, EPOCHS, LEARNING_RATE, TRAINING_BATCH);
    nn_test(nn, "training", TRAINING_IMAGES_COUNT, IMAGE_UNIT_LEN, training_images, training_labels);

    nn_save(nn, SAVED_MODELS_PATH, LEARNING_RATE, EPOCHS);

    dataset_load("test", test_images, test_labels);
    nn_test(nn, "test", TEST_IMAGES_COUNT, IMAGE_UNIT_LEN, test_images, test_labels);
  }

  else if ((argc == 3) && (strcmp(argv[1], "-test") == 0)) {
    char *model_name = argv[2];
    nn_load(nn, SAVED_MODELS_PATH, model_name);
    
    dataset_load("training", training_images, training_labels);
    nn_test(nn, "training", TRAINING_IMAGES_COUNT, IMAGE_UNIT_LEN, training_images, training_labels);

    dataset_load("test", test_images, test_labels);
    nn_test(nn, "test", TEST_IMAGES_COUNT, IMAGE_UNIT_LEN, test_images, test_labels);
  }

  else if ((argc == 5) && (strcmp(argv[1], "-guess") == 0) && (strcmp(argv[3], "-model") == 0)) {
    char *model_name = argv[4];
    nn_load(nn, SAVED_MODELS_PATH, model_name);

    char *guess_image_path = argv[2];
    pmg_load(guess_image_path, nn);

    nn_guess(nn);
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

