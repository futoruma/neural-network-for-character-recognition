#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#define OLIVEC_IMPLEMENTATION
#include "olive.c"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"

#define IMG_WIDTH 3240
#define IMG_HEIGHT 3240

uint32_t img_pixels[IMG_WIDTH * IMG_HEIGHT];

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

void nn_save(NN nn, char *save_path, float learning_rate, size_t epochs)
{
  char fullname[128];
  FILE *fptr;

  time_t current_time = time(NULL);
  char date_string[20];
  strftime(date_string, 20, "%Y%m%d%H%M%S", localtime(&current_time));

  strcpy(fullname, save_path);
  strcat(fullname, date_string);
  strcat(fullname, "_784x48x10_");

  char learning_rate_string[20];
  sprintf(learning_rate_string, "%f", learning_rate);
  strcat(fullname, learning_rate_string);
  strcat(fullname, "x");

  char epochs_string[20];
  sprintf(epochs_string, "%zu", epochs);
  strcat(fullname, epochs_string);

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
/*
void nn_render(Olivec_Canvas img, NN nn)
{
  uint32_t background_color = 0xFF181818;
  uint32_t low_color = 0x000000FF;
  uint32_t high_color = 0x00FFFF00;
  olivec_fill(img, background_color);

  int neuron_radius = 25;
  int layer_border_vpad = 50;
  int layer_border_hpad = 50;
  int nn_width = img.width - (layer_border_hpad * 2);
  int nn_height = img.height - (layer_border_vpad * 2);
  int nn_x = (img.width / 2) - (nn_width / 2);
  int nn_y = (img.height / 2) - (nn_height / 2);
  size_t layer_count = nn.count + 1;
  int layer_hpad = nn_width / layer_count;
  for (size_t l = 0; l < layer_count; l++) {
    int layer_vpad1 = nn_height / nn.as[l].cols;  
    for (size_t i = 0; i < nn.as[l].cols; i++) {
      int cx1 = nn_x + (layer_hpad * l) + (layer_hpad / 2);
      int cy1 = nn_y + (layer_vpad1 * i) + (layer_vpad1 / 2);
      if ((l + 1) < layer_count) {
        int layer_vpad2 = nn_height / nn.as[l+1].cols;
        for (size_t j = 0; j < nn.as[l+1].cols; j++) {
          int cx2 = nn_x + (layer_hpad * (l + 1)) + (layer_hpad / 2);
          int cy2 = nn_y + (layer_vpad2 * j) + (layer_vpad2 / 2);
          uint32_t alpha = floorf(sigmoidf(MATRIX_AT(nn.ws[l], i, j)) * 255.0f);
          uint32_t connection_color = 0xFF000000 | low_color;
          olivec_blend_color(&connection_color, (alpha<<(8*3)) | high_color);
          olivec_line(img, cx1, cy1, cx2, cy2, connection_color);
        }   
      }
      if (l > 0) {
          uint32_t alpha = floorf(sigmoidf(MATRIX_AT(nn.bs[l-1], 0, i)) * 255.0f);
          uint32_t neuron_color = 0xFF000000 | low_color;
          olivec_blend_color(&neuron_color, (alpha<<(8*3)) | high_color);
          olivec_circle(img, cx1, cy1, neuron_radius, neuron_color);
      } else {
          olivec_circle(img, cx1, cy1, neuron_radius, 0xFFAAAAAA);
      }
    }
  }
}
*/
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

/*    
    nn_save(nn, SAVED_MODELS_PATH, LEARNING_RATE, EPOCHS);

    printf("Model has been trained and saved.\n");

    nn_test(nn, "training", TRAINING_IMAGES_NUM, IMAGE_UNIT_LEN, 
            training_images, training_labels);
    */
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

    for (size_t i = 0; i < 784; i++) {
      MATRIX_AT(NN_INPUT(nn), 0, i) = (float) buf[i] / 255.0f;
    }

    nn_guess(nn);

    printf("Calculated probabilities:\n");
    for (size_t i = 0; i < DIGITS; i++) {
      printf("(%zu) -> %f%%\n", i, MATRIX_AT(NN_OUTPUT(nn), 0, i) * 100);
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

    Olivec_Canvas img = olivec_canvas(img_pixels, IMG_WIDTH, IMG_HEIGHT, IMG_WIDTH);
  
    nn_render(img, nn);
    char img_file_path[256];
    snprintf(img_file_path, sizeof(img_file_path), "./render/%s.png", argv[2]);

    if (!stbi_write_png(img_file_path, img.width, img.height, 4, img_pixels, img.stride * sizeof(uint32_t))) {
      printf("ERROR: could not save file %s\n", img_file_path);
    }
  }

  else {
    printf("Invalid parameters.\n");
  }

  return 0;
}

