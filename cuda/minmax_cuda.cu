#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <float.h>
#include <cuda_runtime.h>

// Error checking macro for CUDA calls
#define CHECK_CUDA(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", \
                    __FILE__, __LINE__, cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Kernel 1: Find minimum value in array
__global__ void find_min_kernel(const float* data, float* min_vals, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory with large value
    sdata[tid] = FLT_MAX;
    
    // Each thread processes multiple elements (grid stride loop)
    while (i < n) {
        if (data[i] < sdata[tid]) {
            sdata[tid] = data[i];
        }
        i += blockDim.x * gridDim.x;
    }
    
    __syncthreads();
    
    // Parallel reduction in shared memory with optimized loop
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] < sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Final warp reduction (no __syncthreads needed)
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64 && smem[tid + 32] < smem[tid]) smem[tid] = smem[tid + 32];
        if (blockDim.x >= 32 && smem[tid + 16] < smem[tid]) smem[tid] = smem[tid + 16];
        if (smem[tid + 8] < smem[tid]) smem[tid] = smem[tid + 8];
        if (smem[tid + 4] < smem[tid]) smem[tid] = smem[tid + 4];
        if (smem[tid + 2] < smem[tid]) smem[tid] = smem[tid + 2];
        if (smem[tid + 1] < smem[tid]) smem[tid] = smem[tid + 1];
    }
    
    // Write block's min to global memory
    if (tid == 0) {
        min_vals[blockIdx.x] = sdata[0];
    }
}

// Kernel 2: Find maximum value in array
__global__ void find_max_kernel(const float* data, float* max_vals, int n) {
    extern __shared__ float sdata[];
    
    int tid = threadIdx.x;
    int i = blockIdx.x * blockDim.x + tid;
    
    // Initialize shared memory with small value
    sdata[tid] = -FLT_MAX;
    
    // Each thread processes multiple elements (grid stride loop)
    while (i < n) {
        if (data[i] > sdata[tid]) {
            sdata[tid] = data[i];
        }
        i += blockDim.x * gridDim.x;
    }
    
    __syncthreads();
    
    // Parallel reduction in shared memory with optimized loop
    for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
        if (tid < s) {
            if (sdata[tid + s] > sdata[tid]) {
                sdata[tid] = sdata[tid + s];
            }
        }
        __syncthreads();
    }
    
    // Final warp reduction (no __syncthreads needed)
    if (tid < 32) {
        volatile float* smem = sdata;
        if (blockDim.x >= 64 && smem[tid + 32] > smem[tid]) smem[tid] = smem[tid + 32];
        if (blockDim.x >= 32 && smem[tid + 16] > smem[tid]) smem[tid] = smem[tid + 16];
        if (smem[tid + 8] > smem[tid]) smem[tid] = smem[tid + 8];
        if (smem[tid + 4] > smem[tid]) smem[tid] = smem[tid + 4];
        if (smem[tid + 2] > smem[tid]) smem[tid] = smem[tid + 2];
        if (smem[tid + 1] > smem[tid]) smem[tid] = smem[tid + 1];
    }
    
    // Write block's max to global memory
    if (tid == 0) {
        max_vals[blockIdx.x] = sdata[0];
    }
}

// Kernel 3: Normalize array using min-max scaling
__global__ void normalize_kernel(float* data, int n, float min_val, float max_val) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    
    // Grid stride loop for better GPU utilization
    while (i < n) {
        float range = max_val - min_val;
        if (range != 0.0f) {
            data[i] = (data[i] - min_val) / range;
        } else {
            data[i] = 0.5f;  // All values are equal
        }
        i += blockDim.x * gridDim.x;
    }
}

// Generate random data on host
void generate_random_data(float* data, int n) {
    srand(time(NULL));
    for (int i = 0; i < n; i++) {
        data[i] = (float)rand() / RAND_MAX * 100.0f;
    }
}

// Find global min from block results on CPU
float find_global_min(float* block_mins, int num_blocks) {
    float global_min = FLT_MAX;
    for (int i = 0; i < num_blocks; i++) {
        if (block_mins[i] < global_min) {
            global_min = block_mins[i];
        }
    }
    return global_min;
}

// Find global max from block results on CPU
float find_global_max(float* block_maxs, int num_blocks) {
    float global_max = -FLT_MAX;
    for (int i = 0; i < num_blocks; i++) {
        if (block_maxs[i] > global_max) {
            global_max = block_maxs[i];
        }
    }
    return global_max;
}

// Validate GPU results with CPU reference
void validate_results(float* gpu_data, float* cpu_data, int n) {
    int errors = 0;
    float tolerance = 1e-5f;
    
    for (int i = 0; i < n; i++) {
        if (fabs(gpu_data[i] - cpu_data[i]) > tolerance) {
            errors++;
            if (errors <= 5) {
                printf("  Mismatch at [%d]: GPU=%.6f, CPU=%.6f\n", 
                       i, gpu_data[i], cpu_data[i]);
            }
        }
    }
    
    if (errors == 0) {
        printf("  Validation: PASSED (all values match within tolerance)\n");
    } else {
        printf("  Validation: FAILED (%d mismatches found)\n", errors);
    }
}

int main(int argc, char* argv[]) {
    if (argc != 4) {
        printf("Usage: %s <array_size> <block_size> <num_blocks>\n", argv[0]);
        printf("Example: %s 1000000 256 256\n", argv[0]);
        return 1;
    }
    
    // Parse arguments
    int n = atoi(argv[1]);
    int block_size = atoi(argv[2]);
    int num_blocks = atoi(argv[3]);
    
    if (n <= 0 || block_size <= 0 || num_blocks <= 0) {
        printf("All parameters must be positive\n");
        return 1;
    }
    
    printf("CUDA Min-Max Normalization\n");
    printf("Array size: %d\n", n);
    printf("Block size: %d\n", block_size);
    printf("Number of blocks: %d\n", num_blocks);
    printf("Total threads: %d\n", block_size * num_blocks);
    printf("\n");
    
    // Allocate host memory
    float* h_data = (float*)malloc(n * sizeof(float));
    float* h_data_copy = (float*)malloc(n * sizeof(float));
    float* h_normalized = (float*)malloc(n * sizeof(float));
    
    if (!h_data || !h_data_copy || !h_normalized) {
        printf("Host memory allocation failed\n");
        return 1;
    }
    
    // Generate random data
    generate_random_data(h_data, n);
    memcpy(h_data_copy, h_data, n * sizeof(float));
    
    // Allocate device memory
    float* d_data;
    float* d_block_mins;
    float* d_block_maxs;
    
    CHECK_CUDA(cudaMalloc(&d_data, n * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_block_mins, num_blocks * sizeof(float)));
    CHECK_CUDA(cudaMalloc(&d_block_maxs, num_blocks * sizeof(float)));
    
    // Copy data to device
    CHECK_CUDA(cudaMemcpy(d_data, h_data, n * sizeof(float), cudaMemcpyHostToDevice));
    
    // Prepare timing variables
    cudaEvent_t start, stop;
    CHECK_CUDA(cudaEventCreate(&start));
    CHECK_CUDA(cudaEventCreate(&stop));
    
    float kernel_time = 0.0f;
    float total_time = 0.0f;
    
    // Start total timer
    clock_t cpu_start = clock();
    CHECK_CUDA(cudaEventRecord(start, 0));
    
    // Launch min reduction kernel
    dim3 grid(num_blocks);
    dim3 block(block_size);
    size_t shared_mem_size = block_size * sizeof(float);
    
    find_min_kernel<<<grid, block, shared_mem_size>>>(d_data, d_block_mins, n);
    CHECK_CUDA(cudaGetLastError());
    
    // Launch max reduction kernel
    find_max_kernel<<<grid, block, shared_mem_size>>>(d_data, d_block_maxs, n);
    CHECK_CUDA(cudaGetLastError());
    
    // Copy block results to host
    float* h_block_mins = (float*)malloc(num_blocks * sizeof(float));
    float* h_block_maxs = (float*)malloc(num_blocks * sizeof(float));
    
    CHECK_CUDA(cudaMemcpy(h_block_mins, d_block_mins, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    CHECK_CUDA(cudaMemcpy(h_block_maxs, d_block_maxs, num_blocks * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Find global min and max
    float global_min = find_global_min(h_block_mins, num_blocks);
    float global_max = find_global_max(h_block_maxs, num_blocks);
    
    // Launch normalization kernel
    normalize_kernel<<<grid, block>>>(d_data, n, global_min, global_max);
    CHECK_CUDA(cudaGetLastError());
    
    // Wait for all kernels to complete
    CHECK_CUDA(cudaDeviceSynchronize());
    
    // Stop timer and calculate kernel time
    CHECK_CUDA(cudaEventRecord(stop, 0));
    CHECK_CUDA(cudaEventSynchronize(stop));
    CHECK_CUDA(cudaEventElapsedTime(&kernel_time, start, stop));
    
    // Calculate total CPU time
    clock_t cpu_end = clock();
    double total_cpu_time = ((double)(cpu_end - cpu_start)) / CLOCKS_PER_SEC * 1000.0; // Convert to ms
    
    // Copy normalized data back to host
    CHECK_CUDA(cudaMemcpy(h_normalized, d_data, n * sizeof(float), cudaMemcpyDeviceToHost));
    
    // Calculate CPU reference for validation
    float cpu_min = FLT_MAX;
    float cpu_max = -FLT_MAX;
    
    for (int i = 0; i < n; i++) {
        if (h_data_copy[i] < cpu_min) cpu_min = h_data_copy[i];
        if (h_data_copy[i] > cpu_max) cpu_max = h_data_copy[i];
    }
    
    float* cpu_normalized = (float*)malloc(n * sizeof(float));
    float cpu_range = cpu_max - cpu_min;
    
    if (cpu_range == 0.0f) {
        for (int i = 0; i < n; i++) cpu_normalized[i] = 0.5f;
    } else {
        for (int i = 0; i < n; i++) {
            cpu_normalized[i] = (h_data_copy[i] - cpu_min) / cpu_range;
        }
    }
    
    // Display results
    printf("Results:\n");
    printf("  GPU Min value: %.6f\n", global_min);
    printf("  GPU Max value: %.6f\n", global_max);
    printf("  CPU Min value: %.6f\n", cpu_min);
    printf("  CPU Max value: %.6f\n", cpu_max);
    printf("\n");
    
    // Display timing
    printf("Performance:\n");
    printf("  Kernel execution time: %.3f ms\n", kernel_time);
    printf("  Total GPU time (incl. memcpy): %.3f ms\n", total_cpu_time);
    printf("  Memory bandwidth: %.2f GB/s\n", 
           (3.0f * n * sizeof(float) / (kernel_time / 1000.0f)) / 1e9);
    printf("\n");
    
    // Validate results
    printf("Validation:\n");
    validate_results(h_normalized, cpu_normalized, n);
    printf("\n");
    
    // Display first 5 normalized values
    printf("First 5 normalized values:\n");
    for (int i = 0; i < 5 && i < n; i++) {
        printf("  data[%d] = %.6f\n", i, h_normalized[i]);
    }
    
    // Cleanup
    free(h_data);
    free(h_data_copy);
    free(h_normalized);
    free(h_block_mins);
    free(h_block_maxs);
    free(cpu_normalized);
    
    CHECK_CUDA(cudaFree(d_data));
    CHECK_CUDA(cudaFree(d_block_mins));
    CHECK_CUDA(cudaFree(d_block_maxs));
    CHECK_CUDA(cudaEventDestroy(start));
    CHECK_CUDA(cudaEventDestroy(stop));
    
    // Reset device for clean exit
    CHECK_CUDA(cudaDeviceReset());
    
    return 0;
}