#include <cuda_runtime.h>
#include <cuda.h>
#include <stdio.h>

__global__ void mykernel(int *addr) {
    atomicAdd(addr, 10);       // only available on devices with compute capability 6.x
    printf("%d\n", *addr);
  }
  
void foo() {
    int *addr;
    cudaMallocManaged(&addr, 4);
    *addr = 0;

    mykernel<<<1, 1>>>(addr);
    __sync_fetch_and_add(addr, 10);  // CPU atomic operation
}