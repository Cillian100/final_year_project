#include <stdio.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE 128

__device__ void swap(int* array, int work_id_1, int work_id_2){
  int temp=array[work_id_2];
  array[work_id_2]=array[work_id_1];
  array[work_id_1]=temp;
}

__device__ void print_array_kernel(int* array, int N){
  for(int a=0;a<N;a++){
    printf("%d ", array[a]);
  }
  printf("\n");
}

__host__ void print_array_host(int* array, int N){
  for(int a=0;a<N;a++){
    printf("%d ", array[a]);
  }
  printf("\n");
}

__global__ void odd_even_kernel(int* array, int N){
  int work_id = threadIdx.x;
  if(work_id>=N){
    return;
  }
  
  __syncthreads();
  __shared__ int shared_array[BLOCK_SIZE];
  __shared__ bool isSorted[1];

  if(work_id==0){
    isSorted[0]=false;
  }
  shared_array[work_id]=array[work_id];
  __syncthreads();
  
  while(isSorted[0]==false){
    isSorted[0]=true;

    if(work_id%2==1 && work_id+1!=N){
      if(shared_array[work_id]>shared_array[work_id+1]){
	swap(shared_array, work_id, work_id+1);
	isSorted[0]=false;
      }
    }
    __syncthreads();
    
    if(work_id%2==0 && work_id+1!=N){
      if(shared_array[work_id]>shared_array[work_id+1]){
	swap(shared_array, work_id, work_id+1);
	isSorted[0]=false;
      }
    }
    __syncthreads();
  }
  array[work_id]=shared_array[work_id];
}

__device__ int ceiling_kernel(double number){
  if(number==(int)number){
    return number;
  }else{
    return (int)number+1;
  }
}

__device__ int minimum_kernel(int value1, int value2){
  if(value1<value2){
    return value1;
  }else{
    return value2;
  }
}

__device__ int co_rank(int k, int* A, int m, int* B, int n){
  int i = k < m ? k : m;
  int j = k - i;
  int i_low = 0 > (k-n) ? 0 : k-n;
  int j_low = 0 > (k-m) ? 0 : k-m;
  int delta;
  bool active=true;

  while(active){
    if( (i > 0) && (j < n) && (A[i-1] > B[j]) ){
      delta = ((i - i_low + 1) >> 1);
      j_low = j;
      j = j + delta;
      i = i - delta;
    }else if((j > 0) && (i < m) && (B[j-1]>=A[i])){
      delta = ((j - j_low + 1) >> 1);
      i_low = i;
      i = i + delta;
      j = j - delta;
    }else{
      active=false;
    }
  }
  return i;
}

__device__ void merge_sequential_device(int* A, int m, int* B, int n, int* C){
  int i=0;
  int j=0;
  int k=0;

  while((i<m)&&(j<n)){
    if(A[i]<=B[j]){
      C[k++] = A[i++];
    }else{
      C[k++] = B[j++];
    }
  }

  if(i == m){
    while(j < n){
      C[k++] = B[j++];
    }
  }else{
    while(i < m){
      C[k++] = A[i++];
    }
  }
}

__global__ void merge_kernel(int* A, int m, int* B, int n, int* C){
  int work_id = threadIdx.x + blockDim.x * blockIdx.x;
  if(work_id>=m+n){
    return;
  }

  int elementsPerThread = ceiling_kernel((double)(m+n)/(blockDim.x*gridDim.x));
  int k_curr = work_id*elementsPerThread;
  int k_next = minimum_kernel((work_id+1)*elementsPerThread, m+n);
  int i_curr = co_rank(k_curr, A, m, B, n);
  int i_next = co_rank(k_next, A, m, B, n);
  int j_curr = k_curr - i_curr;
  int j_next = k_next - i_next;

  merge_sequential_device(&A[i_curr], i_next-i_curr, &B[j_curr], j_next-j_curr, &C[k_curr]);
}


int main(){
  int N=BLOCK_SIZE*4;
  int* array_host = nullptr;
  int* array_device = nullptr;
  int* merge_output = nullptr;
  int threadsPerBlock=BLOCK_SIZE;
  int blocksPerGrid=(threadsPerBlock + N - 1)/threadsPerBlock;

  cudaMallocHost(&array_host, N*sizeof(int));
  for(int a=0;a<N;a++){
    array_host[a]=rand()%100;
  }

  cudaMalloc(&array_device, N*sizeof(int));
  cudaMalloc(&merge_output, N*sizeof(int));
  cudaMemcpy(array_device, array_host, N*sizeof(int), cudaMemcpyDefault);

  for(int a=0;a<blocksPerGrid;a++){
    odd_even_kernel<<<1, BLOCK_SIZE>>>(array_device+(BLOCK_SIZE*a), BLOCK_SIZE);
  }
  cudaDeviceSynchronize();
  
  //this line can be removed
  cudaMemcpy(merge_output, array_device, N*sizeof(int), cudaMemcpyDefault);

  merge_kernel<<<2, BLOCK_SIZE>>>(array_device,
				  BLOCK_SIZE,
				  array_device+(BLOCK_SIZE),
				  BLOCK_SIZE,
				  merge_output);
  
  merge_kernel<<<2, BLOCK_SIZE>>>(array_device+(BLOCK_SIZE*2),
  				  BLOCK_SIZE,
  				  array_device+(BLOCK_SIZE*3),
  				  BLOCK_SIZE,
  				  merge_output);

  cudaDeviceSynchronize();
  cudaMemcpy(array_device, merge_output, N*sizeof(int), cudaMemcpyDefault);

  merge_kernel<<<4, BLOCK_SIZE>>>(array_device,
				  BLOCK_SIZE*2,
				  array_device+(BLOCK_SIZE*2),
				  BLOCK_SIZE*2,
				  merge_output);
  
  cudaMemcpy(array_host, merge_output, N*sizeof(int), cudaMemcpyDefault);

  for(int a=0;a<blocksPerGrid;a++){
    for(int b=0;b<BLOCK_SIZE;b++){
      printf("%d ", array_host[b+BLOCK_SIZE*a]);
    }
    printf("\n");
  }

  cudaDeviceSynchronize();
}
