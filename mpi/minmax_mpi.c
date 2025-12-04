#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <mpi.h>

void generate_random_data(float *data, int n, int seed) {
    srand(seed);
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX * 100.0f;
    }
}

void find_local_min_max(float *data, int n, float *local_min, float *local_max) {
    *local_min = FLT_MAX;
    *local_max = -FLT_MAX;
    
    for (int i = 0; i < n; i++) {
        if (data[i] < *local_min) *local_min = data[i];
        if (data[i] > *local_max) *local_max = data[i];
    }
}

void normalize_local_data(float *data, int n, float global_min, float global_max) {
    float range = global_max - global_min;
    
    if (range == 0.0f) {
        for (int i = 0; i < n; i++) {
            data[i] = 0.5f;
        }
        return;
    }
    
    for (int i = 0; i < n; i++) {
        data[i] = (data[i] - global_min) / range;
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    
    int rank, num_procs;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
    
    if (argc != 2) {
        if (rank == 0) {
            printf("Usage: mpirun -np <processes> %s <array_size>\n", argv[0]);
        }
        MPI_Finalize();
        return 1;
    }
    
    int n = atoi(argv[1]);
    if (n <= 0) {
        if (rank == 0) {
            printf("Array size must be positive\n");
        }
        MPI_Finalize();
        return 1;
    }
    
    // Calculate data distribution
    int local_n = n / num_procs;
    int remainder = n % num_procs;
    
    // Determine local size for this process
    int my_count = local_n;
    if (rank < remainder) my_count++;
    
    // Allocate local memory
    float *local_data = (float*)malloc(my_count * sizeof(float));
    
    // Generate random data locally in each process (for fair comparison)
    // Use different seed per process for variety
    generate_random_data(local_data, my_count, 12345 + rank);
    
    // Barrier to ensure all processes start timing together
    MPI_Barrier(MPI_COMM_WORLD);
    
    // Start timing AFTER data generation
    double start_time = MPI_Wtime();
    
    // Find local min and max
    float local_min, local_max;
    find_local_min_max(local_data, my_count, &local_min, &local_max);
    
    // Perform global reduction to find overall min and max
    float global_min, global_max;
    MPI_Allreduce(&local_min, &global_min, 1, MPI_FLOAT, MPI_MIN, MPI_COMM_WORLD);
    MPI_Allreduce(&local_max, &global_max, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);
    
    // Normalize local data
    normalize_local_data(local_data, my_count, global_min, global_max);
    
    // End timing (before any gather operations)
    double end_time = MPI_Wtime();
    double time_taken = end_time - start_time;
    
    // Root process displays results
    if (rank == 0) {
        printf("MPI Min-Max Scaling\n");
        printf("Array size: %d\n", n);
        printf("Number of processes: %d\n", num_procs);
        printf("Global min value: %.4f\n", global_min);
        printf("Global max value: %.4f\n", global_max);
        printf("Time taken: %.6f seconds\n", time_taken);
        
        // Verify first 5 values from rank 0's local data
        printf("\nFirst 5 normalized values (from rank 0):\n");
        for (int i = 0; i < 5 && i < my_count; i++) {
            printf("data[%d] = %.6f\n", i, local_data[i]);
        }
    }
    
    // Clean up local allocations
    free(local_data);
    
    MPI_Finalize();
    return 0;
}