
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <omp.h>

void generate_random_data(float *data, int n) {
    unsigned int seed = time(NULL);
    
    #pragma omp parallel
    {
        unsigned int thread_seed = seed + omp_get_thread_num();
        
        #pragma omp for
        for (int i = 0; i < n; i++) {
            
            thread_seed = thread_seed * 1103515245 + 12345;
            data[i] = ((float)((thread_seed / 65536) % 32768) / 32768.0f) * 100.0f;
        }
    }
}

void find_min_max(float *data, int n, float *min_val, float *max_val) {
    float local_min = FLT_MAX;
    float local_max = -FLT_MAX;
    
    #pragma omp parallel for reduction(min:local_min) reduction(max:local_max)
    for (int i = 0; i < n; i++) {
        if (data[i] < local_min) local_min = data[i];
        if (data[i] > local_max) local_max = data[i];
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
    
    
    omp_set_num_threads(num_threads);
    
    
    float *data = (float*)malloc(n * sizeof(float));
    if (data == NULL) {
        printf("Memory allocation failed\n");
        return 1;
    }
    
    double start_time = omp_get_wtime();
    
    
    generate_random_data(data, n);
    
    
    float min_val, max_val;
    find_min_max(data, n, &min_val, &max_val);
    
    
    minmax_normalize(data, n, min_val, max_val);
    
    double end_time = omp_get_wtime();
    double time_taken = end_time - start_time;
    
    
    printf("OpenMP Min-Max Scaling\n");
    printf("Array size: %d\n", n);
    printf("Number of threads: \033[32m%d\033[0m\n", num_threads);
    printf("Min value: %.4f\n", min_val);
    printf("Max value: %.4f\n", max_val);
    printf("Time taken: \033[32m%.6f\033[0m seconds\n", time_taken);
    
    
    printf("\nFirst 5 normalized values:\n");
    for (int i = 0; i < 5 && i < n; i++) {
        printf("data[%d] = %.6f\n", i, data[i]);
    }
    
    free(data);
    return 0;
}
