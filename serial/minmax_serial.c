#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>

void generate_random_data(float *data, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX * 100.0f; // Values between 0-100
    }
}

void find_min_max(float *data, int n, float *min_val, float *max_val) {
    *min_val = FLT_MAX;
    *max_val = -FLT_MAX;  // Fixed: FLT_MIN is positive minimum
    
    for (int i = 0; i < n; i++) {
        if (data[i] < *min_val) *min_val = data[i];
        if (data[i] > *max_val) *max_val = data[i];
    }
}

void minmax_normalize(float *data, int n, float min_val, float max_val) {
    float range = max_val - min_val;
    if (range == 0.0f) {
        // If all values are same, set to 0.5
        for (int i = 0; i < n; i++) {
            data[i] = 0.5f;
        }
        return;
    }
    
    for (int i = 0; i < n; i++) {
        data[i] = (data[i] - min_val) / range;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 2) {
        printf("Usage: %s <array_size>\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    if (n <= 0) {
        printf("Array size must be positive\n");
        return 1;
    }
    
    // Allocate memory
    float *data = (float*)malloc(n * sizeof(float));
    if (data == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    // Generate data
    clock_t start = clock();
    generate_random_data(data, n);
    
    // Find min and max
    float min_val, max_val;
    find_min_max(data, n, &min_val, &max_val);
    
    // Normalize
    minmax_normalize(data, n, min_val, max_val);
    
    clock_t end = clock();
    double time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;
    
    // Display results
    printf("Serial Min-Max Scaling\n");
    printf("Array size: %d\n", n);
    printf("Min value: %.4f\n", min_val);
    printf("Max value: %.4f\n", max_val);
    printf("Time taken: %.6f seconds\n", time_taken);
    
    // Verify first 5 values
    printf("\nFirst 5 normalized values:\n");
    for (int i = 0; i < 5 && i < n; i++) {
        printf("data[%d] = %.6f\n", i, data[i]);
    }
    
    free(data);
    return 0;
}