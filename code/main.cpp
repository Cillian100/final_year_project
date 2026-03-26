#include <stdio.h>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <algorithm>
#include <pthread.h>
#include <chrono>
#include <cstring>
#include <stdlib.h>
#include <iterator>

extern void tbb_sort(std::vector<long int> &myVec, long int startingPoint, long int endPoint);
extern void cuda_thrust_sort(std::vector<long int> &myVec, int deviceNumber, long int startingPoint, long int endingPoint);

struct myStruct{
	std::vector<long int> &myVector;
	long startingPoint;
	long endingPoint;
	int deviceNumber;
	std::vector<long int> &vector1;
	std::vector<double> &vector2;
};

struct weights{
	float cpuWeight;
	float gpuWeight;
};

void* calling_tbb(void* args){
	myStruct* s = static_cast<myStruct*>(args);
	//printf("Starting TBB Sort. Length %ld\n", s->endingPoint-s->startingPoint);
	auto start = std::chrono::high_resolution_clock::now();
	//tbb_sort(s->myVector, s->startingPoint, s->endingPoint);
	tbb_sort(s->myVector, s->startingPoint, s->endingPoint);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed = end - start;
	//printf("Finishing TBB Sort - elapsed: %.3f ms/n\n\n", elapsed.count());
	s->vector1.push_back(s->endingPoint - s->startingPoint);
	s->vector2.push_back(elapsed.count());
	printf("%ld, %.3f\n", s->endingPoint - s->startingPoint, elapsed.count());
	return NULL;
}

void* calling_thrust(void* args){
	myStruct* s = static_cast<myStruct*>(args);
	//printf("Starting Thrust Sort %ld\n", s->endingPoint - s->startingPoint);
	auto start = std::chrono::high_resolution_clock::now();
	cuda_thrust_sort(s->myVector, s->deviceNumber, s->startingPoint, s->endingPoint);
	auto end = std::chrono::high_resolution_clock::now();
	std::chrono::duration<double, std::milli> elapsed = end - start;
	//printf("Finishing Thrust Sort - elapsed: %.3f ms/n\n\n", elapsed.count());
	
	s->vector2.push_back(elapsed.count());
	printf("%ld, %.3f\n", s->endingPoint - s->startingPoint, elapsed.count());
	return NULL;
}

void testingForCPU(int iter, int forLoopSize, std::vector<long int> &nums, std::vector<long int> &vector1, std::vector<double> &vector2){
	std::vector<long int> nums2;
	int variable=iter;
	for(int a=0;a<forLoopSize;a++){
		nums2.clear();
		pthread_t thread_1;
		copy(nums.begin(), nums.begin()+variable, back_inserter(nums2));
		myStruct struct1 = {nums2, 0, variable, 0, vector1, vector2};
		pthread_create(&thread_1, NULL, calling_tbb, &struct1);
		pthread_join(thread_1, NULL);
		variable=variable+iter;
	}
}

void testingForGPU(int iter, int forLoopSize, std::vector<long int> &nums, std::vector<long int> &vector1, std::vector<double> &vector2){
	std::vector<long int> nums2;	
	long int variable=iter;
	for(int a=0;a<forLoopSize;a++){
		nums2.clear();
		pthread_t thread_1;
		copy(nums.begin(), nums.begin()+variable, back_inserter(nums2));
		myStruct struct1 = {nums2, 0, variable, 1, vector1, vector2};
		pthread_create(&thread_1, NULL, calling_thrust, &struct1);
		pthread_join(thread_1, NULL);
		variable=variable+iter;
	}
}

void linearRegression(std::vector<long int>& x, std::vector<double>& y, double& m, double& b){
	int n = x.size();
	double sumX=0, sumY=0, sumXY=0, sumX2=0;
	
	for(int i=0; i<n; i++){
		sumX += (double)x[i];
		sumY += y[i];
		sumXY += (double)x[i] * y[i];
		sumX2 += (double)x[i] * (double)x[i];
	}
	m = (n*sumXY - sumX*sumY) / (n*sumX2 - sumX*sumX);
	b = (sumY - m*sumX) / n;
}

void runningTests(std::string filename){
	std::ifstream file(filename);
 	printf("reading %s\n", filename.c_str());
	std::vector<long int> nums(std::istream_iterator<long int>(file), {});
    int iter=100000000;
	int forLoopSize=10;
	std::vector<long int> vector1;
	std::vector<double> vector2;
	std::vector<double> vector3;

	printf("CPU Testing\n");
	testingForCPU(iter, forLoopSize, nums, vector1, vector2);
	printf("\n\nGPU Testing\n");
	testingForGPU(iter, forLoopSize, nums, vector1, vector3);

	for(int a=0; a<vector1.size(); a++){
		printf("%ld %f %f\n", vector1.at(a), vector2.at(a), vector3.at(a));
	}
	double test1;
	double test2;
	
	for(int a=0;a<vector1.size();a++){
		vector1.at(a)=vector1.at(a)/10000;
	}

	linearRegression(vector1, vector2, test1, test2);
	printf("\nCPU linear regression weights %f %f\n", test1, test2);
	linearRegression(vector1, vector3, test1, test2);
	printf("\nGPU linear regression weights %f %f\n", test1, test2);
	nums.clear();
}

void measuring_speed(std::string filename){
	std::ifstream file(filename);
	printf("reading %s\n", filename.c_str());
	std::vector<long int> nums(std::istream_iterator<long int>(file), {});
	int iter=1000000;
	int forLoopSize=100;
	std::vector<long int> vector1;
	std::vector<double> vector2;
	std::vector<double> vector3;

}

int main(){
	measuring_speed("dataToSort/averageCase");
}
