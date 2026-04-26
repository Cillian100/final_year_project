#include <iostream>
#include <vector>
#include <algorithm>
#include <numeric>
#include <chrono>
#include <random>
#include <fstream>



void lsdCountingPass(std::vector<int>& arr, int exp) {
    const int n = arr.size();
    std::vector<int> out(n);
    int cnt[10] = {};
    for (int x : arr)             cnt[(x / exp) % 10]++;
    for (int i = 1; i < 10; i++)  cnt[i] += cnt[i - 1];
    for (int i = n - 1; i >= 0; i--)
        out[--cnt[(arr[i] / exp) % 10]] = arr[i];
    arr = out;
}

void lsdRadixSort(std::vector<int>& arr) {
    if (arr.empty()) return;
    int maxVal = *std::max_element(arr.begin(), arr.end());
    for (int exp = 1; maxVal / exp > 0; exp *= 10)
        lsdCountingPass(arr, exp);
}

void msdHelper(std::vector<int>& arr, std::vector<int>& buf,
               int lo, int hi, int exp) {
    if (hi <= lo || exp == 0) return;

    int cnt[10] = {};
    for (int i = lo; i <= hi; i++)
        cnt[(arr[i] / exp) % 10]++;

    int start[10];
    start[0] = lo;
    for (int d = 1; d < 10; d++)
        start[d] = start[d - 1] + cnt[d - 1];

    int pos[10];
    std::copy(start, start + 10, pos);
    for (int i = lo; i <= hi; i++) {
        int d = (arr[i] / exp) % 10;
        buf[pos[d]++] = arr[i];
    }

    for (int i = lo; i <= hi; i++)
        arr[i] = buf[i];

    int nextExp = exp / 10;
    for (int d = 0; d < 10; d++)
        msdHelper(arr, buf, start[d], start[d] + cnt[d] - 1, nextExp);
}

void msdRadixSort(std::vector<int>& arr) {
    if (arr.empty()) return;
    int maxVal = *std::max_element(arr.begin(), arr.end());
    int exp = 1;
    while (maxVal / exp >= 10) exp *= 10;
    std::vector<int> buf(arr.size());
    msdHelper(arr, buf, 0, (int)arr.size() - 1, exp);
}

void mergeHalves(std::vector<int>& arr, int lo, int mid, int hi) {
    std::vector<int> tmp(arr.begin() + lo, arr.begin() + hi + 1);
    int L = 0, R = mid - lo + 1, k = lo;
    int Le = mid - lo, Re = hi - lo;
    while (L <= Le && R <= Re)
        arr[k++] = (tmp[L] <= tmp[R]) ? tmp[L++] : tmp[R++];
    while (L <= Le) arr[k++] = tmp[L++];
    while (R <= Re) arr[k++] = tmp[R++];
}

void mergeSortHelper(std::vector<int>& arr, int lo, int hi) {
    if (lo >= hi) return;
    int mid = lo + (hi - lo) / 2;
    mergeSortHelper(arr, lo, mid);
    mergeSortHelper(arr, mid + 1, hi);
    mergeHalves(arr, lo, mid, hi);
}

void mergeSort(std::vector<int>& arr) {
    if (arr.size() < 2) return;
    mergeSortHelper(arr, 0, (int)arr.size() - 1);
}

void mo3(std::vector<int>& arr, int lo, int hi) {
    int mid = lo + (hi - lo) / 2;
    if (arr[lo]  > arr[mid]) std::swap(arr[lo],  arr[mid]);
    if (arr[lo]  > arr[hi])  std::swap(arr[lo],  arr[hi]);
    if (arr[mid] > arr[hi])  std::swap(arr[mid], arr[hi]);
    std::swap(arr[mid], arr[hi - 1]);   // place pivot at hi-1
}

void quickSortHelper(std::vector<int>& arr, int lo, int hi) {
    if (hi - lo < 2) {
        if (hi > lo && arr[lo] > arr[hi]) std::swap(arr[lo], arr[hi]);
        return;
    }
    mo3(arr, lo, hi);
    int pivot = arr[hi - 1], i = lo, j = hi - 1;
    while (true) {
        while (arr[++i] < pivot) {}
        while (arr[--j] > pivot) {}
        if (i >= j) break;
        std::swap(arr[i], arr[j]);
    }
    std::swap(arr[i], arr[hi - 1]);
    quickSortHelper(arr, lo, i - 1);
    quickSortHelper(arr, i + 1, hi);
}

void quickSort(std::vector<int>& arr) {
    if (arr.size() < 2) return;
    quickSortHelper(arr, 0, (int)arr.size() - 1);
}

void printToFile(std::vector<int> length, std::vector<float> lsdRadixSortArray, 
    std::vector<float> msdRadixSortArray, std::vector<float> mergeSortArray, std::vector<float> quickSortArray, int iterations){
    std::ofstream myFile("filename.txt");

    myFile << "size: ";
    for(int a=0;a<iterations;a++){
        myFile << length.at(a) << ", ";
    }
    myFile << std::endl;

    myFile << "lsdRadixSort: ";
    for(int a=0;a<iterations;a++){
        myFile << lsdRadixSortArray.at(a) << ", ";
    }
    myFile << std::endl;

    myFile << "msdRadixSort: ";
    for(int a=0;a<iterations;a++){
        myFile << msdRadixSortArray.at(a) << ", ";
    }
    myFile << std::endl;

    myFile << "mergeSort: ";
    for(int a=0;a<iterations;a++){
        myFile << mergeSortArray.at(a) << ", ";
    }
    myFile << std::endl;

    myFile << "quickSort: ";
    for(int a=0;a<iterations;a++){
        myFile << quickSortArray.at(a) << ", ";
    }
    myFile << std::endl;
}

int main() {
    int N = 1000;
    int iterations=7;
    std::vector<int> length(iterations);
    std::vector<float> lsdRadixSortArray(iterations);
    std::vector<float> msdRadixSortArray(iterations);
    std::vector<float> mergeSortArray(iterations);
    std::vector<float> quickSortArray(iterations);

    for(int i=0;i<iterations;i++){
        std::cout << N << std::endl;
        std::vector<int> arr(N);
        std::mt19937_64 rng(std::random_device{}());
        std::uniform_int_distribution<int> dist(0, 100000000);
        std::generate(arr.begin(), arr.end(), [&](){
            return (long long)dist(rng);
        });
        double ms; 
        std::vector<int> a = arr;
    
        auto t1 = std::chrono::high_resolution_clock::now();
        lsdRadixSort(a);
        auto t2 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t2 - t1).count();
        std::cout << "lsdRadixSort: " << N / ms << " N/ms" << std::endl;
        lsdRadixSortArray[i]=(N/ms);


        std::vector<int> b = arr;  
        auto t3 = std::chrono::high_resolution_clock::now();
        msdRadixSort(b);
        auto t4 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t4 - t3).count();
        std::cout << "msdRadixSort: " << N / ms << " N/ms" << std::endl;
        msdRadixSortArray[i]=(N/ms);
    
        std::vector<int> c = arr;
        auto t5 = std::chrono::high_resolution_clock::now();
        mergeSort(c);
        auto t6 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t6 - t5).count();
        std::cout << "mergeSort: " << N / ms << " N/ms" << std::endl;
        mergeSortArray[i]=(N/ms);

        std::vector<int> d = arr;  
        auto t7 = std::chrono::high_resolution_clock::now();
        quickSort(d);
        auto t8 = std::chrono::high_resolution_clock::now();
        ms = std::chrono::duration<double, std::milli>(t8 - t7).count();
        std::cout << "quickSort: " << N/ms << " N/ms" << std::endl;
        quickSortArray[i]=(N/ms);

        N=N*10;
    }

    printToFile(length, lsdRadixSortArray, msdRadixSortArray, mergeSortArray, quickSortArray, iterations);
}