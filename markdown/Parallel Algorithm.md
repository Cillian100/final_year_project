### Merge Sort
Follows the divide and conquer approach, works by recursively diving the input array into two halves, recursively sorting the two halves and finally merging them back together to obtain the sorted array.
```c
void merge(int arr[], int left, int middle, int right){
	int i, j, k;
	int n1 = middle - left + 1;
	int n2 = right - middle;
	
	int L[n1], R[n2];
	
	for(i = 0; i < n1; i++){
		L[i] = arr[left + i];
	}
	for(j = 0; j < n2; j++){
		R[j] = arr[middle + 1 + j];
	}
	
	i = 0;
	j = 0;
	k = l;
	
	while(i < n1 && j < n2){
		if(L[i] <= R[j]){
			arr[k] = L[i];
			i++;
		}else{
			arr[k] = R[j];
			j++;
		}
		k++;
	}
	
	while(i < n1){
		arr[k]=L[i];
		i++;
		k++;
	}
	
	while(j < n2){
		arr[k]=R[j];
		j++;
		k++;
	}
}

void mergeSort(int arr[], int left, int right){
	if(left < right){
		int middle = left + (right - left) / 2;
		
		mergeSort(arr, left, middle);
		mergeSort(arr, middle+1, right);
		
		merge(arr, left, middle, right);
	}
}
```
### Bitonic Sort
- bitonic sort is a parallel sorting algorithm designed to take full advantage of hardware that can perform multiple operations simultaneously.
- unlike traditional sorting algorithms bitonic sort is built to exploit parallelism 
- it works only when the number of elements is a power of 2

- bitonic sort works by recursively building bitonic sequences and then merging them into a fully sorted array - this structure allows the algorithm to perform many compare and swap operations simultaneously, which is why its ideal for parallel execution. 
```c
void comp_and_swap(int array[], int i, int j, int direction){
	if((direction==1 && array[i] > array[j]) || (direction==0 && array[i]< array[j])){
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
		
		butonic_merge(array, low, cnt, direction);
	}
}
```
### Odd-Even Sort 
- This is basically a variation of bubble-sort, this algorithm is divided into two phases, odd and even phase. The algorithm runs until the array elements are sorted and in each iteration two phases occur, odd and even phase.
- In the odd phase, we perform a bubble sort on individual elements and in the even phase, we perform a bubble sort on even indexed elements. 
```c
void swap_array(int array[], int one, int two){
	int temp=array[one];
	array[one]=array[two];
	array[two]=temp;
}

void odd_even_sort(int array[], int n){
	bool isSorted = false;
	
	while(!isSorted){
		isSorted=true;
		
		for(int i=1;i<=n-2; i=i+2){
			if(array[i]>array[i+1]){
				swap_array(array, i, i+1);
				isSorted=false;
			}
		}
		
		for(int i=0;i<=n-2;i=i+2){
			if(array[i]>array[i+1]){
				swap_array(array, i, i+1);
				isSorted=false;
			}
		}
	}
}
```

### Quick Sort
- A sorting algorithm based on the divide and conqure that picks and element as a pivot and partitions the given array around the picked pivot by placing the pivot in its correct position in the sorted array.
1. **choose a pivot:** selects an element from the array as the pivot 
2. **partition the array:** re-arrange the array around the pivot, after partitioning all elements smaller than the pivot will be on its left, all elements greater than the pivot will be on its right
3. **recursively call:** recursively apply the same process to the two partitioned subarray
4. **base case:** the recursion stops when there is only one element left in the sub-array

```c
void swap(int *a, int *b){
	int t = *a;
	*a = b*;
	*b = t;
}

int partition(int array[], int low, int high){
	int pivot = array[high];
	
	int i = low - 1;
	
	for(int j=low; j<= high - 1; j++){
		if(array[j] < pivot){
			i++;
			swap(&array[i], &swap[j]);
		}
	}
	
	swap(&array[i + 1], &array[high]);
	
	return i + 1;
}

void quick_sort(int array[], int low, int high){
	if(low < high){
		int pi = parition(array, low, high);
		quick_sort(array, low, pi - 1);
		quick_sort(array, pi + 1, high);
	}
}
```

### Parallel Quick Sort