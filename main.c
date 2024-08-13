#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define TRAINING_IMAGES_PATH "./training_data/training_images"
#define TRAINING_LABELS_PATH "./training_data/training_labels"
#define TEST_IMAGES_PATH "./test_data/test_images"
#define TEST_LABELS_PATH "./test_data/test_labels"

#define TRAINING_IMAGES_NUM 60000
#define TEST_IMAGES_NUM 10000
#define IMAGE_UNIT_LEN 784
#define LABEL_UNIT_LEN 1
#define IMAGE_METADATA_LEN 4
#define LABEL_METADATA_LEN 2

int image_metadata[IMAGE_METADATA_LEN];
int label_metadata[LABEL_METADATA_LEN];

unsigned char training_images_buffer[TRAINING_IMAGES_NUM][IMAGE_UNIT_LEN];
unsigned char test_images_buffer[TEST_IMAGES_NUM][IMAGE_UNIT_LEN];
unsigned char training_labels_buffer[TRAINING_IMAGES_NUM][LABEL_UNIT_LEN];
unsigned char test_labels_buffer[TEST_IMAGES_NUM][LABEL_UNIT_LEN];

float training_images_matrix[TRAINING_IMAGES_NUM][IMAGE_UNIT_LEN];
float test_images_matrix[TEST_IMAGES_NUM][IMAGE_UNIT_LEN];
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
    
  for (int i = 0; i < data_num; i++) {
    read(file_descriptor, buffer[i], unit_len * sizeof(unsigned char));   
  }

  close(file_descriptor);
}


void save_buffer_to_images_matrix(int data_num, unsigned char buffer[][IMAGE_UNIT_LEN], float images_matrix[][IMAGE_UNIT_LEN])
{
  for (int i = 0; i < data_num; i++)
    for (int j = 0; j < IMAGE_UNIT_LEN; j++)
      images_matrix[i][j]  = (float) buffer[i][j] / 255.0f;
}


void save_buffer_to_labels(int data_num, unsigned char buffer[][LABEL_UNIT_LEN], int labels[])
{
  for (int i = 0; i < data_num; i++)
    labels[i]  = (int) buffer[i][0];
}


int main(void)
{  
  load_into_buffer(TRAINING_IMAGES_PATH, IMAGE_METADATA_LEN, image_metadata, IMAGE_UNIT_LEN, TRAINING_IMAGES_NUM, training_images_buffer);
  save_buffer_to_images_matrix(TRAINING_IMAGES_NUM, training_images_buffer, training_images_matrix);

  load_into_buffer(TEST_IMAGES_PATH, IMAGE_METADATA_LEN, image_metadata, IMAGE_UNIT_LEN, TEST_IMAGES_NUM, test_images_buffer);
  save_buffer_to_images_matrix(TEST_IMAGES_NUM, test_images_buffer, test_images_matrix);
    
  load_into_buffer(TRAINING_LABELS_PATH, LABEL_METADATA_LEN, label_metadata, LABEL_UNIT_LEN, TRAINING_IMAGES_NUM, training_labels_buffer);
  save_buffer_to_labels(TRAINING_IMAGES_NUM, training_labels_buffer, training_labels);
    
  load_into_buffer(TEST_LABELS_PATH, LABEL_METADATA_LEN, label_metadata, LABEL_UNIT_LEN, TEST_IMAGES_NUM, test_labels_buffer);
  save_buffer_to_labels(TEST_IMAGES_NUM, test_labels_buffer, test_labels);

  for (int i = 0; i < 784; i++) {
    printf("%1.1f ", test_images_matrix[999][i]);
    if ( (i + 1) % 28 == 0) printf("\n");
  }

  printf("label: %d\n", test_labels[999]);

  return 0;
}

