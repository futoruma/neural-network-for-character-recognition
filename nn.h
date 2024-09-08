#ifndef NN_H_
#define NN_H_

typedef struct {
  size_t rows;
  size_t cols;
  float *items;
} Matrix;

typedef struct {
  size_t count;
  Matrix *ws;
  Matrix *bs;
  Matrix *as;
} NN;

float rand_float(void);
float sigmoidf(float x);

Matrix matrix_alloc(size_t rows, size_t cols);
void matrix_copy(Matrix dst, Matrix src);
void matrix_dot(Matrix dst, Matrix a, Matrix b);
void matrix_fill(Matrix matrix, float x);
void matrix_print(Matrix matrix, const char *name, size_t padding);
void matrix_rand(Matrix matrix, float low, float high);
void matrix_sig(Matrix matrix);
void matrix_softmax(Matrix matrix);
void matrix_sum(Matrix dst, Matrix a);

NN nn_alloc(size_t *arch, size_t arch_count);
void nn_forward(NN nn);
void nn_get_average_gradient(NN gradient, size_t data_count);
void nn_get_total_gradient(NN nn, NN gradient);
void nn_guess(NN nn);
void nn_init(NN nn);
void nn_load(NN nn, char *save_path, char *filename);
void nn_print(NN nn, const char *name);
void nn_render(Olivec_Canvas canvas, NN nn);
void nn_save(NN nn, char *save_path);
void nn_test(NN nn, char *dataset_name, size_t images_count, float images[][IMAGE_UNIT_LEN], int *labels);
void nn_train(NN nn, NN gradient, size_t images_count, float images[][IMAGE_UNIT_LEN], int *labels);
void nn_update_weights(NN nn, NN gradient, float learning_rate);
void nn_zero(NN nn);

#define ARRAY_LEN(xs) sizeof((xs)) / sizeof((xs)[0])
#define NN_INPUT(nn) (nn).as[0]
#define MATRIX_AT(matrix, i, j) (matrix).items[(i) * (matrix).cols + (j)] 
#define MATRIX_PRINT(m) matrix_print(m, #m, 0)
#define NN_OUTPUT(nn) (nn).as[(nn).count]
#define NN_PRINT(nn) nn_print(nn, #nn)

#endif // NN_H_

#ifdef NN_IMPLEMENTATION

float sigmoidf(float x)
{
  return 1.0f / (1.0f + expf(-x));
}

float rand_float(void)
{
  return (float) rand() / (float) RAND_MAX;
}

Matrix matrix_alloc(size_t rows, size_t cols)
{
  Matrix matrix;
  matrix.rows = rows;
  matrix.cols = cols;
  matrix.items = malloc(sizeof(*matrix.items) * rows * cols);
  assert(matrix.items != NULL);
  return matrix;
}

void matrix_copy(Matrix dst, Matrix src)
{
  assert(dst.rows == src.rows);
  assert(dst.cols == src.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MATRIX_AT(dst, i, j) = MATRIX_AT(src, i, j);
    }
  }
}

void matrix_dot(Matrix dst, Matrix a, Matrix b)
{
  assert(a.cols == b.rows);
  size_t n = a.cols;
  assert(dst.rows == a.rows);
  assert(dst.cols == b.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MATRIX_AT(dst, i, j) = 0;
      for (size_t k = 0; k < n; ++k) {
        MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, k) * MATRIX_AT(b, k, j); 
      }
    }
  }
}

void matrix_fill(Matrix matrix, float x)
{
  for (size_t i = 0; i < matrix.rows; ++i) {
    for (size_t j = 0; j < matrix.cols; ++j) {
      MATRIX_AT(matrix, i, j) = x; 
    }
  }
}

void matrix_print(Matrix matrix, const char *name, size_t padding)
{
  printf("%*s%s = [\n", (int) padding, "", name);
  for (size_t i = 0; i < matrix.rows; ++i) {
    printf("%*s ", (int) padding, "");
    for (size_t j = 0; j < matrix.cols; ++j) {
      printf("%f ", MATRIX_AT(matrix, i, j));
    }
    printf("\n");
  }
  printf("%*s]\n", (int) padding, "");
}

void matrix_rand(Matrix matrix, float low, float high)
{
  for (size_t i = 0; i < matrix.rows; ++i) {
    for (size_t j = 0; j < matrix.cols; ++j) {
      MATRIX_AT(matrix, i, j) = rand_float() * (high - low) + low;
    }
  }
}

void matrix_sig(Matrix matrix)
{
  for (size_t i = 0; i < matrix.rows; ++i) {
    for (size_t j = 0; j < matrix.cols; ++j) {
      MATRIX_AT(matrix, i, j) = sigmoidf(MATRIX_AT(matrix, i, j));
    }
  }
}

void matrix_softmax(Matrix matrix)
{
  float max_value = 0;
  for (size_t i = 0; i < matrix.rows; ++i) {
    for (size_t j = 0; j < matrix.cols; ++j) {
      if (MATRIX_AT(matrix, i, j) > max_value) {
        max_value = MATRIX_AT(matrix, i, j);
      }
    }
  }
    
  float exp_sum = 0;
  for (size_t i = 0; i < matrix.rows; ++i) {
    for (size_t j = 0; j < matrix.cols; ++j) {
      MATRIX_AT(matrix, i, j) = expf(MATRIX_AT(matrix, i, j) - max_value);
      exp_sum += MATRIX_AT(matrix, i, j);
    }
  }

  for (size_t i = 0; i < matrix.rows; ++i) {
    for (size_t j = 0; j < matrix.cols; ++j) {
      MATRIX_AT(matrix, i, j) /= exp_sum;
    }
  }
}

void matrix_sum(Matrix dst, Matrix a)
{
  assert(dst.rows == a.rows);
  assert(dst.cols == a.cols);
  for (size_t i = 0; i < dst.rows; ++i) {
    for (size_t j = 0; j < dst.cols; ++j) {
      MATRIX_AT(dst, i, j) += MATRIX_AT(a, i, j);
    }
  }
}

NN nn_alloc(size_t *arch, size_t arch_count)
{
  assert(arch_count > 0);
  NN nn;
  nn.count = arch_count - 1;

  nn.ws = malloc(sizeof(*nn.ws) * nn.count);
  assert(nn.ws != NULL);
  nn.bs = malloc(sizeof(*nn.bs) * nn.count);
  assert(nn.bs != NULL);
  nn.as = malloc(sizeof(*nn.as) * (nn.count + 1));
  assert(nn.as != NULL);

  nn.as[0] = matrix_alloc(1, arch[0]); 
  for (size_t i = 1; i < arch_count; ++i) {
    nn.ws[i-1] = matrix_alloc(nn.as[i-1].cols, arch[i]);
    nn.bs[i-1] = matrix_alloc(1, arch[i]);
    nn.as[i] = matrix_alloc(1, arch[i]);
  }
  return nn;
}

void nn_forward(NN nn)
{
  for (int l = 0; l < nn.count; ++l) {
    matrix_dot(nn.as[l+1], nn.as[l], nn.ws[l]);
    matrix_sum(nn.as[l+1], nn.bs[l]);
    matrix_sig(nn.as[l+1]);
  }
}

void nn_get_average_gradient(NN gradient, size_t data_count)
{
  for (size_t l = 0; l < gradient.count; ++l) {
    for (size_t i = 0; i < gradient.ws[l].rows; ++i) {
      for (size_t j = 0; j < gradient.ws[l].cols; ++j) {
        MATRIX_AT(gradient.ws[l], i, j) /= data_count;
      }
    }
    for (size_t i = 0; i < gradient.bs[l].rows; ++i) {
      for (size_t j = 0; j < gradient.bs[l].cols; ++j) {
        MATRIX_AT(gradient.bs[l], i, j) /= data_count;
      }
    }
  }
}

void nn_get_total_gradient(NN nn, NN gradient)
{
  for (size_t l = nn.count; l > 0; --l) {
    for (size_t j = 0; j < nn.as[l].cols; ++j) {
      float a = MATRIX_AT(nn.as[l], 0, j);
      float da = MATRIX_AT(gradient.as[l], 0, j);
      MATRIX_AT(gradient.bs[l-1], 0, j) += 2 * da * a * (1 - a);
      for (size_t k = 0; k < nn.as[l-1].cols; ++k) {
        float prev_a = MATRIX_AT(nn.as[l-1], 0, k);
        float w = MATRIX_AT(nn.ws[l-1], k, j);
        MATRIX_AT(gradient.ws[l-1], k, j) += 2 * da * a * (1 - a) * prev_a;
        MATRIX_AT(gradient.as[l-1], 0, k) += 2 * da * a * (1 - a) * w;
      }
    }
  }
}

void nn_guess(NN nn)
{
  printf("Guessing the number...\n");
  
  for (size_t l = 0; l < nn.count; ++l) {
    matrix_dot(nn.as[l+1], nn.as[l], nn.ws[l]);
    matrix_sum(nn.as[l+1], nn.bs[l]);
    if ((l + 1) < nn.count)  {
      matrix_sig(nn.as[l+1]);
    }
  }

  matrix_softmax(NN_OUTPUT(nn));

  printf("Calculated probabilities:\n");
  for (size_t i = 0; i < DIGITS; ++i) {
    printf("(%zu) -> %f %%\n", i, MATRIX_AT(NN_OUTPUT(nn), 0, i) * MAX_PERCENT);
  }
}

void nn_init(NN nn)
{
  float limit;
  for (size_t l = 0; l < nn.count; ++l) {
    limit = sqrt(6) / sqrt(nn.as[l].cols + nn.as[l+1].cols);
    matrix_rand(nn.ws[l], -limit, limit);
    matrix_fill(nn.bs[l], 0);
  }
}

void nn_load(NN nn, char *save_path, char *filename)
{
  char fullname[MAX_FILEPATH_LEN];
  strcpy(fullname, save_path);
  strcat(fullname, filename);

  int file_descriptor = open(fullname, O_RDONLY);
  if (file_descriptor == -1) {
    fprintf(stderr, "Error loading the model.");
    exit(1);
  }

  float buf[IMAGE_UNIT_LEN];

  for (size_t l = 0; l < nn.count; ++l) {
    for (size_t i = 0; i < nn.ws[l].rows; ++i) {
      read(file_descriptor, buf, nn.ws[l].cols * sizeof(float));
      for (size_t j = 0; j < nn.ws[l].cols; ++j) {
        MATRIX_AT(nn.ws[l], i, j) = buf[j];
      }
    }
    for (size_t i = 0; i < nn.bs[l].rows; ++i) {
      read(file_descriptor, buf, nn.bs[l].cols * sizeof(float));
      for (size_t j = 0; j < nn.bs[l].cols; ++j) {
        MATRIX_AT(nn.bs[l], i, j) = buf[j];
      }
    }
  }

  close(file_descriptor);
}

void nn_print(NN nn, const char *name)
{
  printf("Printing the model...\n");
  
  size_t padding = 4;
  char buf[256];
  
  printf("%s = [\n", name);
  for (size_t i = 0; i < nn.count; ++i) {
    snprintf(buf, sizeof(buf), "ws%zu", i);
    matrix_print(nn.ws[i], buf, padding);
    snprintf(buf, sizeof(buf), "bs%zu", i);
    matrix_print(nn.bs[i], buf, padding);
  }
  printf("]\n");
}

void nn_render(Olivec_Canvas canvas, NN nn)
{
  printf("Rendering the model...\n");

  uint32_t background_color = BACKGROUND_COLOR;
  uint32_t high_color = HIGH_COLOR;
  uint32_t low_color = LOW_COLOR;
  uint32_t neutral_color = 0xFFAAAAAA;
  uint32_t mask = 0xFF000000;
  olivec_fill(canvas, background_color);

  int neuron_radius = 25;
  int layer_border_vpad = 50;
  int layer_border_hpad = 50;
  int nn_width = canvas.width - (layer_border_hpad * 2);
  int nn_height = canvas.height - (layer_border_vpad * 2);
  int nn_x = (canvas.width / 2) - (nn_width / 2);
  int nn_y = (canvas.height / 2) - (nn_height / 2);
  size_t layer_count = nn.count + 1;
  int layer_hpad = nn_width / layer_count;
  for (size_t l = 0; l < layer_count; ++l) {
    int layer_vpad1 = nn_height / nn.as[l].cols;  
    for (size_t i = 0; i < nn.as[l].cols; ++i) {
      int cx1 = nn_x + (layer_hpad * l) + (layer_hpad / 2);
      int cy1 = nn_y + (layer_vpad1 * i) + (layer_vpad1 / 2);
      if ((l + 1) < layer_count) {
        int layer_vpad2 = nn_height / nn.as[l+1].cols;
        for (size_t j = 0; j < nn.as[l+1].cols; ++j) {
          int cx2 = nn_x + (layer_hpad * (l + 1)) + (layer_hpad / 2);
          int cy2 = nn_y + (layer_vpad2 * j) + (layer_vpad2 / 2);
          uint32_t alpha = floorf(sigmoidf(MATRIX_AT(nn.ws[l], i, j)) * MAX_BRIGHTNESS);
          uint32_t connection_color = mask | low_color;
          olivec_blend_color(&connection_color, (alpha<<(8*3)) | high_color);
          olivec_line(canvas, cx1, cy1, cx2, cy2, connection_color);
        }   
      }
      if (l > 0) {
          uint32_t alpha = floorf(sigmoidf(MATRIX_AT(nn.bs[l-1], 0, i)) * MAX_BRIGHTNESS);
          uint32_t neuron_color = mask | low_color;
          olivec_blend_color(&neuron_color, (alpha<<(8*3)) | high_color);
          olivec_circle(canvas, cx1, cy1, neuron_radius, neuron_color);
      } else {
          olivec_circle(canvas, cx1, cy1, neuron_radius, neutral_color);
      }
    }
  }
  
  printf("The model has been rendered.\n");
}

void nn_save(NN nn, char *save_path)
{
  printf("Saving the model...\n");

  char fullname[MAX_FILEPATH_LEN];
  FILE *fptr;

  time_t current_time = time(NULL);
  char date_string[32];
  strftime(date_string, 32, "%Y%m%d%H%M%S", localtime(&current_time));

  strcpy(fullname, save_path);
  strcat(fullname, date_string);
  strcat(fullname, "_784x48x24x10_");

  char learning_rate_string[32];
  sprintf(learning_rate_string, "%03.0f", LEARNING_RATE * MAX_PERCENT);
  strcat(fullname, learning_rate_string);
  strcat(fullname, "x");

  char epochs_string[32];
  sprintf(epochs_string, "%d", EPOCHS);
  strcat(fullname, epochs_string);

  fptr = fopen(fullname, "wb");
  if ((fptr = fopen(fullname, "wb")) == NULL) {
    fprintf(stderr, "Error opening the file.");
    exit(1);
  }

  float buf[IMAGE_UNIT_LEN];
  for (size_t l = 0; l < nn.count; ++l) {
    for (size_t i = 0; i < nn.ws[l].rows; ++i) {
      for (size_t j = 0; j < nn.ws[l].cols; ++j) {
        buf[j] = MATRIX_AT(nn.ws[l], i, j);
      }
      fwrite(buf, sizeof(float), nn.ws[l].cols, fptr);
    }
    for (size_t i = 0; i < nn.bs[l].rows; ++i) {
      for (size_t j = 0; j < nn.bs[l].cols; ++j) {
        buf[j] = MATRIX_AT(nn.bs[l], i, j);
      }
      fwrite(buf, sizeof(float), nn.bs[l].cols, fptr);
    }
  }

  fclose(fptr);
  printf("The model has been saved.\n");
}

void nn_test(NN nn, char *dataset_name, size_t images_count, float images[][IMAGE_UNIT_LEN], int *labels)
{
  printf("Testing the model...\n");

  size_t correct_count = 0;
  size_t max_digit = 0;
  float max_value = 0.0f;

  for (size_t i = 0; i < images_count; ++i) {
    for (size_t j = 0; j < IMAGE_UNIT_LEN; ++j) {
      MATRIX_AT(NN_INPUT(nn), 0, j) = images[i][j];
    }
    nn_forward(nn);

    max_digit = 0;
    max_value = 0.0f;

    for (size_t d = 0; d < NN_OUTPUT(nn).cols; ++d) {
      if (MATRIX_AT(NN_OUTPUT(nn), 0, d) > max_value) {
        max_value = MATRIX_AT(NN_OUTPUT(nn), 0, d);
        max_digit = d;
      }
    }

    if (max_digit == labels[i]) {
      correct_count++;
    } 
  }
  printf("Forwarded %s set. ", dataset_name);
  printf("Accuracy: %zu / %zu.\n", correct_count, images_count);
}

void nn_train(NN nn, NN gradient, size_t images_count, float images[][IMAGE_UNIT_LEN], int *labels)
{
  printf("Training the model...\n");
  
  srand(time(0));

  nn_init(nn);
  
  Olivec_Canvas canvas = olivec_canvas(canvas_pixels, RENDER_WIDTH, RENDER_HEIGHT, RENDER_WIDTH);
  
  for (size_t e = 0; e < EPOCHS; ++e) {
    nn_zero(gradient);
    
    for (size_t i = 0; i < images_count; ++i) {
      for (size_t j = 0; j < IMAGE_UNIT_LEN; ++j) {
        MATRIX_AT(NN_INPUT(nn), 0, j) = images[i][j];
      }
      nn_forward(nn);

      for (size_t l = 0; l <= nn.count; ++l) {
        matrix_fill(gradient.as[l], 0);
      }

      matrix_copy(NN_OUTPUT(gradient), NN_OUTPUT(nn));
      MATRIX_AT(NN_OUTPUT(gradient), 0, labels[i]) -= MAX_ACTIVATION;

      nn_get_total_gradient(nn, gradient);

      if ((i % TRAINING_BATCH) == (TRAINING_BATCH - 1)) {
        nn_get_average_gradient(gradient, TRAINING_BATCH);
        nn_update_weights(nn, gradient, LEARNING_RATE);
        nn_zero(gradient);
      }
    }

    if (((e % RENDER_STEP) == 9) || e == 0) {
      nn_render(canvas, nn);
      char canvas_filepath[MAX_FILEPATH_LEN];
      snprintf(canvas_filepath, sizeof(canvas_filepath), "./render/%04zu.png", e);

      if (!stbi_write_png(canvas_filepath, canvas.width, canvas.height, PNG_CHANNELS, canvas_pixels, canvas.stride * sizeof(uint32_t))) {
        fprintf(stderr, "Could not save the file.");
      }
    }
  }
  printf("The model has been trained.\n");
}

void nn_update_weights(NN nn, NN gradient, float learning_rate)
{
  for (size_t l = 0; l < gradient.count; ++l) {
    for (size_t i = 0; i < gradient.ws[l].rows; ++i) {
      for (size_t j = 0; j < gradient.ws[l].cols; ++j) {
        MATRIX_AT(nn.ws[l], i, j) -= learning_rate * MATRIX_AT(gradient.ws[l], i, j);
      }
    }
    for (size_t i = 0; i < gradient.bs[l].rows; ++i) {
      for (size_t j = 0; j < gradient.bs[l].cols; ++j) {
        MATRIX_AT(nn.bs[l], i, j) -= learning_rate * MATRIX_AT(gradient.bs[l], i, j);
      }
    }
  }
}

void nn_zero(NN nn)
{
  for (size_t i = 0; i < nn.count; ++i) {
    matrix_fill(nn.ws[i], 0);
    matrix_fill(nn.bs[i], 0);
    matrix_fill(nn.as[i], 0);
  }
  matrix_fill(NN_OUTPUT(nn), 0);
}

#endif // NN_IMPLEMENTATION

