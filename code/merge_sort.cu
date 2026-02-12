#include <iostream>
#include <cuda_runtime.h>

__device__ int co_rank(int k, int* A, int m, int* B, int n){
    int i = k < m ? k : m;
    int j = k - i;
    int i_low = 0 > (k-n) ? 0 : k-n;
    int j_low = 0 > (k-m) ? 0 : k-m;
    int delta;
    bool active = true;

    while(active){
        if(i > 0 && j < n && A[i-1] > B[j]){
            delta = ((i - i_low + 1) >> 1);
            j_low = j;
            j = j + delta;
            i = i - delta;
        }else if(j > 0 && i < m && B[j-1]>=A[i]){
            delta = ((j-j_low+1) >> 1);
            i_low = i;
            i = i + delta;
            j = j - delta;
        }else{
            active=false;
        }
    }

    return i;
}

__device__ void merge_sequential(int *A, int m, int *B, int n, int *C){
    int i = 0;
    int j = 0;
    int k = 0;

    while((i<m) && (j<n)){
        if(A[i]<=B[j]){
            C[k++]=A[i++];
        }else{
            C[k++]=B[j++];
        }
    }
    if(i==m){
        while(j<n){
            C[k++]=B[j++];
        }
    }else{
        while(i<m){
            C[k++]=A[i++];
        }
    }
}

__global__ void merge_basic_kernel(int* A, int m, int* B, int n, int* C){
    int tid = threadIdx.x + blockDim.x * blockIdx.x;
    int totalThreads = blockDim.x * gridDim.x;
    int elementsPerThread = (m+n + totalThreads - 1) / totalThreads;
    int k_curr = tid*elementsPerThread;
    int k_next = fmin((tid+1)*elementsPerThread, m+n);
    int i_curr = co_rank(k_curr, A, m, B, n);
    int i_next = co_rank(k_next, A, m, B, n);
    int j_curr = k_curr - i_curr;
    int j_next = k_next - i_next;
    merge_sequential(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);
}


int main(){
    int N = 10;
    int A[]={1,3,5,7,9};
    int B[]={2,4,6,8,10};
    int C[10];

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;    

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, 5 * sizeof(int));
    cudaMalloc(&d_B, 5 * sizeof(int));
    cudaMalloc(&d_C, 10 * sizeof(int));

    cudaMemcpy(d_A, A, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, 5 * sizeof(int), cudaMemcpyHostToDevice);

    merge_basic_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, 5, d_B, 5, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int a=0;a<N;a++){
        printf("%d\n", C[a]);
    }
    return 0;
}