#include <cuda_runtime.h>
#include <stdio.h>
#include <unistd.h>
#include <pthread.h>

void bitonic_sort_CPU(int* data, int arrayLength){
  for(int k=2; k<=arrayLength; k*=2){
    for(int j=k>>1; j>0; j>>=1){
      for(int i=0;i<arrayLength;i++){
	int ixj = i ^ j;
	
	if(ixj > i){
	  int a = data[i];
	  int b = data[ixj];
	  if((i & k) == 0){
	    if(a > b){
	      data[i] = b;
	      data[ixj] = a;
	    }
	  }else{
	    if(a < b){
	      data[i] = b;
	      data[ixj] = a;
	    }
	  }
	}
      }
    }
  }
}

__global__ void bitonic_sort_GPU(int* data, int j, int k, int arrayLength) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if(i>=arrayLength){
    return;
  }
  int ixj = i ^ j; 
  
  if (ixj > i) {
    int a = data[i];
    int b = data[ixj];
    if ( (i & k) == 0) {
      if (a > b) {
	data[i] = b;
	data[ixj] = a;
      }
    } else {
      if (a < b) {
	data[i] = b;
	data[ixj] = a;
      }
    }
  }
}

typedef struct{
  int* myArray;
  int* deviceArray;
  int arrayLength;
  int deviceValue;
}BitonicArgs;

void* calling_bitonic(void* arg){
  BitonicArgs* args = (BitonicArgs*)arg;

  cudaSetDevice(args->deviceValue);
  for(int k=2; k<=args->arrayLength;k*=2){
    for(int j=k>>1; j>0; j>>=1){
      printf("GPU %d\n", args->deviceValue);
      bitonic_sort_GPU<<<2, 8>>>(args->deviceArray, j, k, args->arrayLength);
    }
  }

  cudaDeviceSynchronize();
  return NULL;
}

void sort_on_CPU(int* myArray, int start, int mid, int end){
  int start2 = mid + 1;

  while(start <= mid && start2 <= end){
    if(myArray[start] <= myArray[start2]){
      start++;
    }else{
      int value = myArray[start2];
      int index = start2;


      while(index!=start){
	myArray[index] = myArray[index - 1];
	index--;
      }

      myArray[start] = value;

      start++;
      mid++;
      start2++;
    }
  }
}


int main(){
  int myArray[]={1, 52, 63, 12, 7, 123, 65, 12, 43, 12, 52, 76, 12, 1, 65, 98, 53, 54};
  int arrayLength=16;
  int half = arrayLength/2;
  int *deviceArray[2];
  pthread_t thread1, thread2;

  cudaSetDevice(0);
  cudaMalloc(&deviceArray[0], half*sizeof(int));
  cudaMemcpy(deviceArray[0], myArray, half*sizeof(int), cudaMemcpyDefault);
  cudaSetDevice(1);
  cudaMalloc(&deviceArray[1], half*sizeof(int));
  cudaMemcpy(deviceArray[1], myArray+half, half*sizeof(int), cudaMemcpyDefault);
  
  BitonicArgs args0 = {myArray, deviceArray[0], half, 0};
  BitonicArgs args1 = {myArray, deviceArray[1], half, 1};

  pthread_create(&thread1, NULL, calling_bitonic, &args0);
  pthread_create(&thread2, NULL, calling_bitonic, &args1);
  pthread_join(thread1, NULL);
  pthread_join(thread2, NULL);
    
  cudaMemcpy(myArray, deviceArray[0], half*sizeof(int), cudaMemcpyDefault);
  cudaMemcpy(myArray+half, deviceArray[1], half*sizeof(int), cudaMemcpyDefault);

  sort_on_CPU(myArray, 0, half, arrayLength);
  
  for(int a=0;a<arrayLength;a++){
    printf("%d ", myArray[a]);
  }
  printf("\n");
}
