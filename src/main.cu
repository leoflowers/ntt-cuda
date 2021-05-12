#include <cmath>		/* pow() */
#include <cstdint>		/* uint64_t */
#include <ctime>		/* time() */

#include <unistd.h>
#include <iostream>


#include "../include/ntt.h"		/* naiveNTT(), outOfPlaceNTT_DIT() */
#include "../include/utils.h"
#include "../include/nttkernel.cuh"	/* inPlaceNTT_DIT_parallel() */

#define ITERS 100

using namespace std;

int main(int argc, char *argv[]){
	uint64_t n = 4096;
  	uint64_t p = 68719403009;
  	uint64_t r = 36048964756;

	clock_t start, end;

	uint64_t vec[n];

  	for (int i = 0; i < n; i++){
    		vec[i] = i;
  	}



  	uint64_t *outVec, *outVecGPU;
       	
	std::cout << "CPU implementation of NTT\n";
	start = clock();
	outVec = inPlaceNTT_DIT(vec, n, p, r, 0);
	end = clock();
	std::cout << "    Non-parallel time: " << (float)(end-start)/CLOCKS_PER_SEC << " s\n";
	
	std::cout << "GPU implementation of NTT\n";
	start = clock();
	outVecGPU = inPlaceNTT_DIT_parallel(vec, n, p, r, 0);
	end = clock();
	std::cout << "    Non-parallel time: " << (float)(end-start)/CLOCKS_PER_SEC << " s\n";

	bool res = compVec(outVec, outVecGPU, n, 1);
	std::cout << "\n\tVectors match?: ";
	if(res) { std::cout << "yes\n"; }
	else { std::cout << "no\n"; }


	return 0;
}
