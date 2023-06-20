#include <stdio.h>
#include <cuda_runtime.h>


__global__ void compute(float* a, float* b, float* c){
    //计算c = a * b
    int position = blockDim.x * blockIdx.x + threadIdx.x;
    c[position] = a[position] * b[position];
}

int main(){
    //定义三个数组，host指针
    const int num = 3;
    float a[num] = {1, 2, 3};
    float b[num] = {5, 7, 9};
    float c[num] = {0};

    //定义3个设备指针，device指针
    size_t size_array = sizeof(c);
    float* device_a = nullptr;
    float* device_b = nullptr;
    float* device_c = nullptr;

    //分配设备空间，大小事size_array,单位byte
    cudaMalloc(&device_a, size_array);
    cudaMalloc(&device_b, size_array);
    cudaMalloc(&device_c, size_array);

    //把数据从host复制到device，其实就是主机复制到显卡
    //复制的是a   b
    cudaMemcpy(device_a, a, size_array, cudaMemcpyHostToDevice);
    cudaMemcpy(device_b, b, size_array, cudaMemcpyHostToDevice);

   //执行核函数，结果放到c上
    // compute<<<1, 3>>>(device_a, device_b, device_c); //注意问这里启动了几个线程 ：答  1 * 3 = 3个
    compute<<<dim3(1), dim3(3)>>>(device_a, device_b, device_c);
        //<<<dim3(1), dim3(3)>>>告诉GPU启动的线程数  ： = 1*1*1*3*1*1=3
     // 第一个  dim3(1)  表示 gridDim 为 1*1*1
     // 第二个  dim3(3)  表示 blockDim 为 3*1*1


    //把计算后的c的结果复制回主机
    cudaMemcpy(c, device_c, size_array, cudaMemcpyDeviceToHost);

    //查看主机上的c内容是多少
    for(int i = 0; i<num;++i){
        printf("c[%d] = %f\n", i, c[i]);
    }

    return 0;
}