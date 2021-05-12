#include <cmath>		/* pow() */
#include <cstdint>		/* uint64_t */
#include <ctime>		/* time() */

#include <unistd.h>
#include <iostream>

#include "../include/ntt.h"		/* naiveNTT(), outOfPlaceNTT_DIT() */
#include "../include/utils.h"
#include "../include/nttkernel.cuh"	/* inPlaceNTT_DIT_parallel() */

using namespace std;

int main(int argc, char *argv[]){
  uint64_t n = 4096;
  uint64_t p = 68719403009;
  uint64_t r = 36048964756;

	uint64_t vec[n];

  for (int i = 0; i < n; i++){
    vec[i] = 0;
  }

  uint64_t *outVec = inPlaceNTT_DIT(vec, n, p, r, 0);
  uint64_t *outVecGPU = inPlaceNTT_DIT_parallel(vec, n, p, r, 0);


	std::cout << "Original vector:\n";
    //printVec(vec, n);
	std::cout << std::endl;	
 
  	std::cout << "CPU result:\n";
	//printVec(outVec, n);
	std::cout << std::endl;

	std::cout << "GPU result:\n";
	printVec(outVecGPU, n);
	std::cout << std::endl;

	  bool res = compVec(outVec, outVecGPU, n, 0);
	  if(res) { std::cout << "vectors match\n"; }


	return 0;

}
