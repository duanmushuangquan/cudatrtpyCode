#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <device_launch_parameters.h>

__global__ void compute(float* a, float* b, float* c){
    int d0 = gridDim.z;
    int d1 = gridDim.y;
    int d2 = gridDim.x;
    int d3 = blockDim.z;
    int d4 = blockDim.y;
    int d5 = blockDim.x;

    int p0 = blockIdx.z;
    int p1 = blockIdx.y;
    int p2 = blockIdx.x;
    int p3 = threadIdx.z;
    int p4 = threadIdx.y;
    int p5 = threadIdx.x;

    int position = (((((p0 * d1) + p1) * d2 + p2) * d3 + p3) * d4 + p4) * d5 + p5;
    c[position] = a[position] * b[position];

    printf("gridDim=%dx%dx%dx, blockDim = %dx%dx%d, [blockIdx = %d,%d,%d, threadIdx = %d,%d,%d], position = %d, avalue = %f\n",
            gridDim.x, gridDim.y, gridDim.z,
            blockDim.x, blockDim.y, blockDim.z,
            blockIdx.x, blockIdx.y, blockIdx.z,
            threadIdx.x, threadIdx.y, threadIdx.z,
            position, a[position]);
}

int main(){
    cudaProfilerStart();

    const int num = 16;
    float a[num] = {1, 2, 3};
    float b[num] = {5, 7, 9};
    float c[num] = {0};

    for(int i=0; i<num;++i){
        a[i] = i;
        b[i] = i;
    }
    size_t size_array = sizeof(c);
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;


    cudaMalloc(&device_a, size_array);
    cudaMalloc(&device_b, size_array);
    cudaMalloc(&device_c, size_array);

    cudaMemcpy(device_a, a, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, size_array, cudaMemcpyHostToDevice);

    compute<<<dim3(1, 2, 2), dim3(2, 2)>>>(device_a, device_b, device_c);

    cudaMemcpy(c, device_c, size_array, cudaMemcpyDeviceToHost);
    for(int i = 0; i< 16;++i){
        printf("c[%d] = %f\n", i, c[i]);
    }

    return 0;
}
