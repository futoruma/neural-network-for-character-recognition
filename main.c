#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
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
#define EPOCHS 1000 
#define LEARNING_RATE 0.03f
#define TRAINING_BATCH 100

int int_buf[IMAGES_METADATA_LEN];
unsigned char char_buf[TRAINING_IMAGES_COUNT][IMAGE_UNIT_LEN];

float training_images[TRAINING_IMAGES_COUNT][IMAGE_UNIT_LEN];
float test_images[TEST_IMAGES_COUNT][IMAGE_UNIT_LEN];
int training_labels[TRAINING_IMAGES_COUNT];
int test_labels[TEST_IMAGES_COUNT];

void load_into_buffer(char *file_path, int meta_len, int unit_len, int data_num)
{
  int file_descriptor = open(file_path, O_RDONLY);
  if (file_descriptor == -1) {
    fprintf(stderr, "Error opening the file.");
    exit(-1);
  }
  
  read(file_descriptor, int_buf, meta_len * sizeof(int));
    
  for (size_t i = 0; i < data_num; ++i) {
    read(file_descriptor, char_buf[i], unit_len * sizeof(unsigned char));   
  }

  close(file_descriptor);
}

void populate_images(int data_num, float images[][IMAGE_UNIT_LEN])
{
  for (size_t i = 0; i < data_num; ++i)
    for (size_t j = 0; j < IMAGE_UNIT_LEN; ++j)
      images[i][j]  = (float) char_buf[i][j] / 255.0f;
}

void populate_labels(int data_num, int labels[])
{
  for (size_t i = 0; i < data_num; ++i)
    labels[i] = (int) char_buf[i][0];
}


int main(int argc, char *argv[])
{ 
  if (argc == 1) {
    printf("Please specify parameters.\n");
    return 0;
  }

  size_t arch[] = {IMAGE_UNIT_LEN, 48, 24, DIGITS};
  size_t layer_count = ARRAY_LEN(arch);

  NN nn = nn_alloc(arch, layer_count);

  if ((argc == 2) && (strcmp(argv[1], "-train") == 0)) {
    printf("Training model...\n");
    
    load_into_buffer(TRAINING_IMAGES_PATH, IMAGES_METADATA_LEN, 
                     IMAGE_UNIT_LEN, TRAINING_IMAGES_COUNT);
    populate_images(TRAINING_IMAGES_COUNT, training_images);
    load_into_buffer(TRAINING_LABELS_PATH, LABELS_METADATA_LEN, 
                     LABEL_UNIT_LEN, TRAINING_IMAGES_COUNT);
    populate_labels(TRAINING_IMAGES_COUNT, training_labels);
    
    NN gradient = nn_alloc(arch, layer_count);

    srand(time(0));

    nn_train(nn, gradient, TRAINING_IMAGES_COUNT, IMAGE_UNIT_LEN, training_images,
             training_labels, EPOCHS, LEARNING_RATE, TRAINING_BATCH);

    nn_save(nn, SAVED_MODELS_PATH, LEARNING_RATE, EPOCHS);

    printf("Model has been trained and saved.\n");

    nn_test(nn, "training", TRAINING_IMAGES_COUNT, IMAGE_UNIT_LEN, 
            training_images, training_labels);
  }

  else if ((argc == 3) && (strcmp(argv[1], "-test") == 0)) {
    printf("Testing saved model...\n");
    
    load_into_buffer(TEST_IMAGES_PATH, IMAGES_METADATA_LEN, 
                     IMAGE_UNIT_LEN, TEST_IMAGES_COUNT);
    populate_images(TEST_IMAGES_COUNT, test_images);
      
    load_into_buffer(TEST_LABELS_PATH, LABELS_METADATA_LEN, 
                     LABEL_UNIT_LEN, TEST_IMAGES_COUNT);
    populate_labels(TEST_IMAGES_COUNT, test_labels);

    load_into_buffer(TRAINING_IMAGES_PATH, IMAGES_METADATA_LEN, 
                     IMAGE_UNIT_LEN, TRAINING_IMAGES_COUNT);
    populate_images(TRAINING_IMAGES_COUNT, training_images);

    load_into_buffer(TRAINING_LABELS_PATH, LABELS_METADATA_LEN, 
                     LABEL_UNIT_LEN, TRAINING_IMAGES_COUNT);
    populate_labels(TRAINING_IMAGES_COUNT, training_labels);

    nn_load(nn, SAVED_MODELS_PATH, argv[2]);
    
    nn_test(nn, "training", TRAINING_IMAGES_COUNT, IMAGE_UNIT_LEN, 
            training_images, training_labels);

    nn_test(nn, "test", TEST_IMAGES_COUNT, IMAGE_UNIT_LEN, 
            test_images, test_labels);
  }

  else if ((argc == 5) && (strcmp(argv[1], "-guess") == 0) &&
           (strcmp(argv[3], "-model") == 0)) {
    printf("Guessing the number...\n");
    
    nn_load(nn, SAVED_MODELS_PATH, argv[4]);

    int file_descriptor = open(argv[2], O_RDONLY);
    if (file_descriptor == -1) {
      fprintf(stderr, "Error opening the file.");
      exit(-1);
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
    printf("Printing saved model...\n");

    nn_load(nn, SAVED_MODELS_PATH, argv[2]);

    NN_PRINT(nn);
  }

  else if ((argc == 3) && (strcmp(argv[1], "-render") == 0)) {
    printf("Rendering saved model...\n");

    nn_load(nn, SAVED_MODELS_PATH, argv[2]);

    Olivec_Canvas canvas = olivec_canvas(canvas_pixels, RENDER_WIDTH, RENDER_HEIGHT, RENDER_WIDTH);
  
    nn_render(canvas, nn);

    char canvas_filepath[256];
    snprintf(canvas_filepath, sizeof(canvas_filepath), "./render/%s.png", argv[2]);

    if (!stbi_write_png(canvas_filepath, canvas.width, canvas.height, 4, canvas_pixels, canvas.stride * sizeof(uint32_t))) {
      printf("ERROR: could not save file %s\n", canvas_filepath);
    }
  }

  else {
    printf("Invalid parameters.\n");
  }

  return 0;
}

