#include <stdio.h>
#include <cmath>
#include <string>
#include <cstdlib>

int main(){
	long startingValue=1000000000;
	std::string filename = "dataToSort/worstCase";
	FILE *f = fopen(filename.c_str(), "w");
	if(!f){
		printf("Failed to open %s\n", filename.c_str());
		return -1;
	}

	for(long int b=startingValue; b>0; b--){
		fprintf(f, "%ld ", b);
	}
	printf("Generated worst data\n");
	fclose(f);

	filename = "dataToSort/averageCase";
	f = fopen(filename.c_str(), "w");
	if(!f){
		printf("Failed to open %s\n", filename.c_str());
		return -1;
	}

	for(long int b=0; b<startingValue; b++){
		fprintf(f, "%d ", rand());
	}
	printf("Generated average data\n");
	fclose(f);
	
	filename = "dataToSort/bestCase";
	f = fopen(filename.c_str(), "w");
	if(!f){
		printf("Failed to open %s\n", filename.c_str());
		return -1;
	}
	
	for(long int b=0; b<startingValue; b++){
		fprintf(f, "%ld ", b);
	}
	printf("Generated best data\n");
	fclose(f);
}
