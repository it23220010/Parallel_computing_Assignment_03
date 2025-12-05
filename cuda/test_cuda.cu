#include <stdio.h>
#include <cuda_runtime.h>

int main() {
    cudaError_t err;
    int deviceCount;
    
    err = cudaGetDeviceCount(&deviceCount);
    printf("cudaGetDeviceCount returned: %d\n", err);
    if (err != cudaSuccess) {
        printf("Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("CUDA devices found: %d\n", deviceCount);
    
    if (deviceCount > 0) {
        cudaDeviceProp prop;
        err = cudaGetDeviceProperties(&prop, 0);
        if (err != cudaSuccess) {
            printf("Error getting properties: %s\n", cudaGetErrorString(err));
            return 1;
        }
        printf("Device 0: %s\n", prop.name);
        printf("Compute capability: %d.%d\n", prop.major, prop.minor);
        
        // Try a simple malloc
        float *d_test;
        err = cudaMalloc(&d_test, 100 * sizeof(float));
        printf("cudaMalloc returned: %d\n", err);
        if (err == cudaSuccess) {
            printf("cudaMalloc succeeded!\n");
            cudaFree(d_test);
        } else {
            printf("cudaMalloc failed: %s\n", cudaGetErrorString(err));
        }
    }
    
    return 0;
}
