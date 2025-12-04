
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <omp.h>

void generate_random_data(float *data, int n) {
    srand(time(NULL));
    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < n; i++) {
            data[i] = (float)rand() / RAND_MAX * 100.0f;
        }
    }
}

void find_min_max(float *data, int n, float *min_val, float *max_val) {
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    
    #pragma omp parallel reduction(min:local_min) reduction(max:local_max)
    {
        float private_min = FLT_MAX;
        float private_max = -FLT_MAX;
        
        #pragma omp for
        for (int i = 0; i < n; i++) {
            if (data[i] < private_min) private_min = data[i];
            if (data[i] > private_max) private_max = data[i];
        }
        
        if (private_min < local_min) local_min = private_min;
        if (private_max > local_max) local_max = private_max;
    }
    
    *min_val = local_min;
    *max_val = local_max;
}

void minmax_normalize(float *data, int n, float min_val, float max_val) {
    float range = max_val - min_val;
    
    if (range == 0.0f) {
        #pragma omp parallel for
        for (int i = 0; i < n; i++) {
            data[i] = 0.5f;
        }
        return;
    }
    
    #pragma omp parallel for
    for (int i = 0; i < n; i++) {
        data[i] = (data[i] - min_val) / range;
    }
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        printf("Usage: %s <array_size> <num_threads>\n", argv[0]);
        return 1;
    }
    
    int n = atoi(argv[1]);
    int num_threads = atoi(argv[2]);
    
    if (n <= 0 || num_threads <= 0) {
        printf("Array size and thread count must be positive\n");
        return 1;
    }
    
    // Set number of threads
    omp_set_num_threads(num_threads);
    
    // Allocate memory
    float *data = (float*)malloc(n * sizeof(float));
    if (data == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    double start_time = omp_get_wtime();
    
    // Generate data
    generate_random_data(data, n);
    
    // Find min and max
    float min_val, max_val;
    find_min_max(data, n, &min_val, &max_val);
    
    // Normalize
    minmax_normalize(data, n, min_val, max_val);
    
    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;
    
    // Display results
    printf("OpenMP Min-Max Scaling\n");
    printf("Array size: %d\n", n);
    printf("Number of threads: %d\n", num_threads);
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