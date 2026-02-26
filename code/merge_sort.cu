#include <iostream>
#include <cuda_runtime.h>
#include <bits/stdc++.h>

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

__device__ inline int min_int(int a, int b) {
    return a < b ? a : b;
}


__global__ void merge_tiled_kernel(int* A, int m, int* B, int n, int* C, int tile_size){
    extern __shared__ int shareAB[];
    int* A_S = &shareAB[0];
    int* B_S = &shareAB[tile_size];
    
    int* shared_corank = &shareAB[2 * tile_size];
    
    int C_curr = blockIdx.x * ((m + n + gridDim.x - 1) / gridDim.x);
    int C_next = min_int((blockIdx.x + 1) * ((m + n + gridDim.x - 1) / gridDim.x), (m + n));

    if(threadIdx.x == 0){
        shared_corank[0] = co_rank(C_curr, A, m, B, n);
        shared_corank[1] = co_rank(C_next, A, m, B, n);
    }
    __syncthreads();
    
    int A_curr = shared_corank[0];
    int A_next = shared_corank[1];
    int B_curr = C_curr - A_curr;
    int B_next = C_next - A_next;

    int C_length = C_next - C_curr;
    int A_length = A_next - A_curr;
    int B_length = B_next - B_curr;
    int total_iteration = (C_length + tile_size - 1) / tile_size;
    int C_completed = 0;
    int A_consumed = 0;
    int B_consumed = 0;

    for(int counter = 0; counter < total_iteration; counter++){
        for(int i = 0; i < tile_size; i += blockDim.x){
            if(i + threadIdx.x < A_length - A_consumed){
                A_S[i + threadIdx.x] = A[A_curr + A_consumed + i + threadIdx.x];
            }
        }
        
        for(int i = 0; i < tile_size; i += blockDim.x){
            if(i + threadIdx.x < B_length - B_consumed){
                B_S[i + threadIdx.x] = B[B_curr + B_consumed + i + threadIdx.x];
            }
        }
        
        __syncthreads();

        int c_curr = threadIdx.x * (tile_size / blockDim.x);
        int c_next = (threadIdx.x + 1) * (tile_size / blockDim.x);
        c_curr = (c_curr <= C_length - C_completed) ? c_curr : C_length - C_completed;
        c_next = (c_next <= C_length - C_completed) ? c_next : C_length - C_completed;
        
        int a_curr = co_rank(c_curr, A_S, min_int(tile_size, A_length - A_consumed), 
                            B_S, min_int(tile_size, B_length - B_consumed));
        int b_curr = c_curr - a_curr;
        
        int a_next = co_rank(c_next, A_S, min_int(tile_size, A_length - A_consumed), 
                            B_S, min_int(tile_size, B_length - B_consumed));
        int b_next = c_next - a_next;

        merge_sequential(A_S + a_curr, a_next - a_curr, B_S + b_curr, b_next - b_curr,
                        C + C_curr + C_completed + c_curr);
        
        __syncthreads();
        
        C_completed += tile_size;
        int merged_this_iteration = min_int(tile_size, C_length - C_completed);

        A_consumed += co_rank(merged_this_iteration,
                     A_S, min_int(tile_size, A_length - A_consumed), 
                     B_S, min_int(tile_size, B_length - B_consumed));
        B_consumed = C_completed - A_consumed;

        __syncthreads();
    }
}

int main(){
    int N = 10;
    int A[]={1,3,5,7,9};
    int B[]={2,4,6,8,10};
    int C[10];

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int tile_size = 128;
    
    size_t shared_mem_size = (2 * tile_size + 2) * sizeof(int);

    int *d_A, *d_B, *d_C;

    cudaMalloc(&d_A, 5 * sizeof(int));
    cudaMalloc(&d_B, 5 * sizeof(int));
    cudaMalloc(&d_C, 10 * sizeof(int));

    cudaMemcpy(d_A, A, 5 * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, 5 * sizeof(int), cudaMemcpyHostToDevice);

    merge_basic_kernel<<<blocksPerGrid, threadsPerBlock, shared_mem_size>>>(d_A, 5, d_B, 5, d_C);
    cudaDeviceSynchronize();

    cudaMemcpy(C, d_C, 10 * sizeof(int), cudaMemcpyDeviceToHost);
    
    for(int a=0;a<N;a++){
        printf("%d\n", C[a]);
    }
    
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    
    return 0;
}