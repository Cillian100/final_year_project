#include <stdio.h>
#include <cstdlib>
#include <stdlib.h>

int getMax(int array[], int n){
    int mx = array[0];
    for(int i=1; i<n; i++){
        if(array[i]>mx){
            mx=array[i];
        }
    }
    return mx;
}

void countSort(int array[], int n, int exp){
    int* output = (int*)malloc(n * sizeof(int));
    int count[10]={0};

    for(int i=0; i<n; i++){
        count[(array[i]/exp) % 10]++;
    }

    for(int i=1; i<10; i++){  // Start at 1, not 0!
        count[i]+=count[i - 1];
    }

    for(int i=n-1; i>= 0; i--){
        output[count[(array[i]/exp) % 10] - 1] = array[i];
        count[(array[i] / exp) % 10]--;
    }

    for(int i=0;i<n;i++){
        array[i]=output[i];
    }
}

void radixSort(int array[], int n){
    int m = getMax(array, n);

    for(int exp=1; m/exp>0; exp*=10){
        countSort(array, n, exp);
    }
}

int main(){
    int array[]={170, 45, 75, 90, 802, 24, 2, 66};
    int n = sizeof(array)/sizeof(array[0]);

    radixSort(array, n);

    for(int a=0;a<n;a++){
        printf("%d ", array[a]);
    }
    printf("\n");
}