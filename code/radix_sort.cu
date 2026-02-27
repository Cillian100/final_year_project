#include <cuda_runtime.h>
#include <stdio.h>

#define RADIX 10
#define BLOCK_SIZE 256

int getMax(int array[], int n){
  int mx = array[0];

  for(int i=1; i<n; i++){
    if(array[i]>mx){
      mx = array[i];
    }
  }

  return mx;
}

void count_sort(int array[], int n, int exp){
  int output[n];
  int count[10] = {0};

  for(int i=0;i<n;i++){
    count[(array[i] / exp) % 10]++;
  }

  for(int i=1; i<10; i++){
    count[i] += count[i - 1];
  }

  for(int i=n-1; i>=0; i--){
    output[count[(array[i]/exp)%10]-1] = array[i];
    count[(array[i]/exp)%10]--;
  }

  for(int i=0; i<n; i++){
    array[i]=output[i];
  }
  
}

void radix_sort_CPU(int array[], int n){
  int m = getMax(array, n);

  for(int exp=1; m/exp>0;exp*=10){
    count_sort(array, n, exp);
  }  
}

__global__ void countKernel(int* array, int n, int exp, int *block_count){
  __shared__ int localCount[RADIX];

  int tid = threadIdx.x;
  int gid = blockIdx.x * blockDim.x + threadIdx.x;

  if(tid < RADIX){
    localCount[tid]=0;
  }
  __syncthreads();

  if(gid < n){
    int digit = (array[gid] / exp) % RADIX;
    atomicAdd(&localCount[digit], 1);
  }

  __syncthreads();


  if(tid < RADIX){
    block_count[blockIdx.x * RADIX + tid] = localCount;
  }
}

void prefix_scan_host(int *blockCounts, int numBlocks, int* globalOffset){
}

void radix_sort_GPU(int host_array[]){
  int *d_array;
  int *d_output;
  int *d_block_count;
  int *d_offset;
  int numbBlock=(n + BLOCK_SIZE - 1) / BLOCK_SIZE;

  cudaMalloc(&d_array, n*sizeof(int));
  cudaMalloc(&d_output, n*sizeof(int));
  cudaMalloc(&d_block_count, numBlocks*RADIX*sizeof(int));
  cudaMalloc(&d_offset, RADIX*sizeof(int));

  cudaMemcpy(device_array, host_array, n*sizeof(int), cudaMemcpyDefault);

  int max = host_array[0];
  for(int i=1;i<n;i++){
    if(host_array[i]>max){
      max=host_array;
    }
  }

  //one pass for digit position
  for(int exp=1; max/exp>0; exp*=RADIX){
    countKernel<<<numBlocks, BLOCK_SIZE>>>(d_array, n, exp, d_block_counts);
    cudaDeviceSynchronize();

    //prefix scan on the CPU
    //TODO rewrite this for the GPU
    int host_block_count[numBlocks * RADIX];
    cudaMemcpy(hosy_block_count, d_block_counts, numBlocks*RADIX*sizeof(int), cudaMemcpyDefault);
    int host_offset[RADIX];
    prefixScanHost(h_block_count, numBlocks, host_offset);
  }
}

int main(){
  int array[] = {170, 45, 75, 90, 802, 24, 2, 66};
  int n = sizeof(array) / sizeof(array[0]);

  //radix_sort_CPU(array, n);
  radix_sort_GPU(array, n);

  for(int a=0;a<n;a++){
    printf("%d\n", array[a]);
  }
  

  return 0;
}
