#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

void swap_array(int array[], int one, int two){
  int temp=array[one];
  array[one]=array[two];
  array[two]=temp;
}

void odd_even_sort(int array[], int n){
  bool isSorted = false;

  while(!isSorted){
    isSorted=true;

    for(int i=1; i<=n-2; i=i+2){
      if(array[i]>array[i+1]){
        swap_array(array, i, i+1);
        isSorted = false;
      }
    }

    for(int i=0; i<=n-2; i=i+2){
      if(array[i] > array[i+1]){
        swap_array(array, i, i+1);
        isSorted = false;
      }
    }
  }
}

int main(){
  int array[]={20, 10, 43, 23};
  int size=sizeof(array)/sizeof(array[0]);
  odd_even_sort(array, size);

  for(int a=0;a<size;a++){
    printf("%d ", array[a]);
  }
  printf("\n");
}
