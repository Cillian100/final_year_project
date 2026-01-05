#include <stdlib.h>
#include <stdio.h>

void print_array(int array[], int low, int high){
  for(int a=low; a<=high; a++){
    printf("%d ", array[a]);
  }

  printf("\n");
}

void comp_and_swap(int array[], int i, int j, int direction){
  if((direction==1 && array[i]>array[j]) || (direction==0 && array[j]>array[i])){
    int temp=array[i];
    array[i]=array[j];
    array[j]=temp;
  }
}

void bitonic_merge(int array[], int low, int count, int direction){
  if(count > 1){
    int k = count / 2;
    
    for(int i=low; i<low+k; i++){
      comp_and_swap(array, i, i+k, direction);
    }

    bitonic_merge(array, low, k, direction);
    bitonic_merge(array, low + k, k, direction);
  }
}

void bitonic_sort(int array[], int low, int count, int direction){
  if(count > 1){
    int k = count / 2;

    bitonic_sort(array, low, k, 1);
    bitonic_sort(array, low + k, k, 0);

    bitonic_merge(array, low, count, direction);
  }
}

int main(){
  int array[]={19, 2, 72, 3, 18, 57, 603, 101};

  int array_size = sizeof(array) / sizeof(array[0]);

  bitonic_sort(array, 0, array_size, 1);
  
  print_array(array, 0, array_size-1);
}
