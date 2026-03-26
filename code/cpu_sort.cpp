#include <oneapi/tbb/parallel_sort.h>
#include <vector>

void hello_from_tbb(){
	printf("hello from tbb\n");
}

void tbb_sort(std::vector<long int>& myVector, long int startingPoint, long int endingPoint){
	tbb::parallel_sort(myVector.begin()+startingPoint, myVector.begin()+endingPoint);
}

void onedpl_sort(std::vector<long int>& myVector, long int startingPoint, long int endingPoint){
	oneapi::dpl::sort(oneapi::dpl::execution::par_unseq, 
			myVector.begin()+startingPoint, 
			myVector.begin()+endingPoint);
}