#include <stdio.h>
#include <stdlib.h>

void comp_and_swap(int array[], int i, int j, int direction){
  if((direction==1 && array[i] > array[j]) || (direction==0 && array[i] < array[j])){
    int temp=array[i];
    array[i]=array[j];
    array[j]=temp;
  }
}

void bitonic_merge(int array[], int low, int cnt, int direction){
  if(cnt > 1){
    int k = cnt / 2;
    for(int i=low; i<low+k; i++){
      comp_and_swap(array, i, i + k, direction);
    }

    bitonic_merge(array, low, k, direction);
    bitonic_merge(array, low + k, k, direction);
  }
}

void bitonic_sort(int array[], int low, int cnt, int direction){
  if(cnt > 1){
    int k = cnt / 2;

    bitonic_sort(array, low, k, 1);
    bitonic_sort(array, low + k, k, 0);

    bitonic_merge(array, low, cnt, direction);
  }
}


int main(){
  int array[] = {10, 3, 4, 8, 6, 2, 1, 5};
  int array_size = sizeof(array) / sizeof(array[0]);
  
  bitonic_sort(array, 0, array_size, 1);

  for(int a=0;a<array_size;a++){
    printf("%d ", array[a]);
  }
  printf("\n");
}
