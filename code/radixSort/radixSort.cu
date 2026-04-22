#include <cuda_runtime.h>
#include <stdio.h>
#include <thrust/device_ptr.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>

__global__ void radix_sort(int* input, int* output, int* bits, int* keys, unsigned int N, unsigned int iter){
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    int key, bit;
    if(i<N){
        key = input[i];
        keys[i] = key;
        bit = (key >> iter) & 1;
        bits[i]=bit;
    }

}

int main(){
    int N=10;
    int *input;
    int *output;
    int *bits;
    int *input2 = new int[N];
    int *keys;
    
    cudaMalloc(&input, N * sizeof(int));
    cudaMalloc(&output, N * sizeof(int));
    cudaMalloc(&bits, N * sizeof(int));
    cudaMalloc(&keys, N * sizeof(int));
    
    for(int a=0;a<N;a++){
        input2[a]=rand()%10;
    }
    cudaMemcpy(input, input2, N * sizeof(int), cudaMemcpyDefault);
    cudaDeviceSynchronize();

    radix_sort<<<2, 10>>>(input, output, bits, keys, N, 0);
    cudaDeviceSynchronize();
    cudaSetDevice(0);
    thrust::exclusive_scan(thrust::device, bits, bits + N, bits);

    cudaMemcpy(input2, input, N * sizeof(int), cudaMemcpyDefault);

    for(int a=0;a<N;a++){
        printf("%d \n", input2[a]);
    }
}

// extractBits<<<blocks, threads>>>(d_input, d_bits, d_keys, N, iter);

// thrust::device_ptr<int> bits_ptr(d_bits);                         // wrap raw pointer
// thrust::exclusive_scan(bits_ptr, bits_ptr + N, bits_ptr);         // ✅ in-place scan

// scatter<<<blocks, threads>>>(d_output, d_bits, d_keys, N);