#include <iostream>
#include <vector>
#include <algorithm>
#include <cmath>
#include <bits/stdc++.h> 
#include <chrono>



int getDigit(int num, int exp) {
    return (num / exp) % 10;
}
 
void countingSortByDigit(std::vector<int>& arr, int exp) {
    int n = arr.size();
    std::vector<int> output(n);
    std::vector<int> count(10, 0);
 
    for (int i = 0; i < n; i++){
        count[getDigit(arr[i], exp)]++;
    }

    for (int i = 1; i < 10; i++){
        count[i] += count[i - 1];
    }

    for (int i = n - 1; i >= 0; i--) {
        int digit = getDigit(arr[i], exp);
        output[count[digit] - 1] = arr[i];
        count[digit]--;
    }
 
    arr = output;
}
 
void lsdRadixSort(std::vector<int>& arr) {
    if (arr.empty()) return;
 
    int maxVal = *std::max_element(arr.begin(), arr.end());
 
    for (int exp = 1; maxVal / exp > 0; exp *= 10)
        countingSortByDigit(arr, exp);
}

int getDigitMSD(int num, int exp) {
    return (num / exp) % 10;
}
 
// Compute the exponent corresponding to the most significant digit of maxVal
int getMSDExp(int maxVal) {
    int exp = 1;
    while (maxVal / exp >= 10)
        exp *= 10;
    return exp;
}
 
// Recursive MSD helper — sorts arr[lo..hi] based on digit at position 'exp'
void msdRadixSortHelper(std::vector<int>& arr, int lo, int hi, int exp) {
    // Base cases: single element, empty range, or no more digit positions
    if (lo >= hi || exp == 0) return;
 
    // Count occurrences of each digit in this range
    std::vector<int> count(10, 0);
    for (int i = lo; i <= hi; i++)
        count[getDigitMSD(arr[i], exp)]++;
 
    // Compute starting index of each bucket
    std::vector<int> bucketStart(10, lo);
    for (int i = 1; i < 10; i++)
        bucketStart[i] = bucketStart[i - 1] + count[i - 1];
 
    // Scatter elements into a temporary output array
    std::vector<int> output(hi - lo + 1);
    std::vector<int> pos = bucketStart; // working copy of bucket positions
    for (int i = lo; i <= hi; i++) {
        int digit = getDigitMSD(arr[i], exp);
        output[pos[digit] - lo] = arr[i];
        pos[digit]++;
    }
 
    // Copy back
    for (int i = lo; i <= hi; i++)
        arr[i] = output[i - lo];
 
    // Recursively sort each bucket on the next less-significant digit
    int nextExp = exp / 10;
    for (int d = 0; d < 10; d++) {
        int bucketLo = bucketStart[d];
        int bucketHi = bucketLo + count[d] - 1;
        if (count[d] > 1)
            msdRadixSortHelper(arr, bucketLo, bucketHi, nextExp);
    }
}
 
void msdRadixSort(std::vector<int>& arr) {
    if (arr.empty()) return;
 
    int maxVal = *std::max_element(arr.begin(), arr.end());
    int exp    = getMSDExp(maxVal);
 
    msdRadixSortHelper(arr, 0, arr.size() - 1, exp);
}
 

void merge(std::vector<int>& arr, int left, int mid, int right){        
    int n1 = mid - left + 1;
    int n2 = right - mid;

    std::vector<int> L(n1), R(n2);

    for (int i = 0; i < n1; i++){
        L[i] = arr[left + i];
    }
    for (int j = 0; j < n2; j++){
        R[j] = arr[mid + 1 + j];
    }

    int i = 0, j = 0;
    int k = left;

    while (i < n1 && j < n2) {
        if (L[i] <= R[j]) {
            arr[k] = L[i];
            i++;
        }
        else {
            arr[k] = R[j];
            j++;
        }
        k++;
    }

    while (i < n1) {
        arr[k] = L[i];
        i++;
        k++;
    }

    while (j < n2) {
        arr[k] = R[j];
        j++;
        k++;
    }
}

void mergeSort(std::vector<int>& arr, int left, int right){
    
    if (left >= right)
        return;

    int mid = left + (right - left) / 2;
    mergeSort(arr, left, mid);
    mergeSort(arr, mid + 1, right);
    merge(arr, left, mid, right);
}

void swap(int *a, int *b){
    int temp=*a;
    *a=*b;
    *b=temp;
}

int partition(std::vector<int>& arr, int low, int high) {  
    int pivot = arr[high];
  
    int i = low - 1;

    for (int j = low; j <= high - 1; j++) {
        if (arr[j] < pivot) {
            i++;
            swap(&arr[i], &arr[j]);
        }
    }
    
    swap(&arr[i + 1], &arr[high]);  
    return i + 1;
}

void quickSort(std::vector<int>& arr, int low, int high) {
      if (low < high) {
        int pi = partition(arr, low, high);

        quickSort(arr, low, pi - 1);
        quickSort(arr, pi + 1, high);
    }
}

 

void printArray(const std::vector<int>& arr, const std::string& label){
    std::cout << label << ": ";
    for(int a=0;a<arr.size();a++){
        printf("%d ", arr[a]);
    }
    std::cout << std::endl;
}

int main(){
    std::vector<int> input = {170, 45, 75, 90, 802, 24, 2, 66, 543, 1, 999, 300};
    double time_taken;

    printf("lsdRadixSort - msdRadixSort - mergeSort - quickSort\n");
    int constant=1000000;
    int n=constant;
    for(int a=0;a<50;a++){
        n=constant*a;
        std::vector<int> randomVector(n);
        for(int a=0;a<n;a++){
            randomVector.at(a)=rand()%10000;
        }

        printf("%d ", n);

        std::vector<int> lsdArray = randomVector;
        auto start = std::chrono::high_resolution_clock::now();
        lsdRadixSort(lsdArray);
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double, std::milli> elapsed = end - start;
        printf("%.5f - ", elapsed.count());

        std::vector<int> msdArray = randomVector;
        start = std::chrono::high_resolution_clock::now();
        msdRadixSort(msdArray);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printf("%.5f - ", elapsed.count());
    
        std::vector<int> mergeArray = randomVector;
        start = std::chrono::high_resolution_clock::now();
        mergeSort(mergeArray, 0, randomVector.size()-1);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printf("%.5f - ", elapsed.count());

        std::vector<int> quickArray = randomVector;
        start = std::chrono::high_resolution_clock::now();
        quickSort(quickArray, 0, randomVector.size()-1);
        end = std::chrono::high_resolution_clock::now();
        elapsed = end - start;
        printf("%.5f\n", elapsed.count());
    }
}