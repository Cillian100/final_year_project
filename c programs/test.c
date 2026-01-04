#include <stdlib.h>
#include <stdio.h>

void print_array(int array[], int low, int high){
  for(int a=low; a<=high; a++){
    printf("%d ", array[a]);
  }
  printf("\n");
}

void merge(int array[], int low, int middle, int high){
  int n1=middle+1-low;
  int n2=high-middle;
  int L[n1], R[n2];
  printf("%d %d\n", n1, n2);

  for(int a=0;a<n1;a++){
    L[a]=array[low+a];
  }
  for(int a=0;a<n2;a++){
    R[a]=array[middle+1+a];
  }

  int pointer1=0, pointer2=0, pointer3=low;

  while(pointer1 < n1 && pointer2 < n2){
    if(L[pointer1] < R[pointer2]){
      array[pointer3]=L[pointer1];
      pointer1++;
    }else{
      array[pointer3]=R[pointer2];
      pointer2++;
    }
    pointer3++;
  }

  while(pointer1 < n1){
    array[pointer3]=L[pointer1];
    pointer1++;
    pointer3++;
  }

  while(pointer2 < n2){
    array[pointer3]=R[pointer2];
    pointer2++;
    pointer3++;
  }
}

void merge_sort(int array[], int low, int high){
  if(low<high){
    int middle=(low+high)/2;

    merge_sort(array, low, middle);
    merge_sort(array, middle+1, high);

    merge(array, low, middle, high);
  }
}


int main(){
  int array[]={10, 14, 63, 23, 61, 24, 76, 29, 65, 3};

  int array_size = sizeof(array) / sizeof(array[0]);
  merge_sort(array, 0, array_size-1);
  
  //merge(array, 0, 2, 4);
  print_array(array, 0, array_size-1);



}
