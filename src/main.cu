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

int main(int argc, char *argv[]) {
	uint64_t n = 4096;
  	uint64_t p = 68719403009;
  	uint64_t r = 36048964756;

	clock_t start, end;

	uint64_t vec[n];

  	for (int i = 0; i < n; i++){
    		vec[i] = i;
  	}



  	uint64_t *outVec, *outVecGPU, *outVecGPU1;
 	int c = log2(n);
	
	const uint64_t *table = offlineModExp(r, p, n);
	/*for(int i = 0; i < c; i++) {
		for(int j = 0; j < c; j++) {
			uint64_t curr = *(table + i*c + j);
			std::cout << curr << " ";
			if(j == 0) {
				std::cout << "(actual: " << modExp(r, (p-1)/pow(2, i), p) << ")  ";
			}
			else {
				std::cout << "(actual: " << modExp( modExp(r, pow(2, i), p), j-1, p) << ")  ";
			}
		}
		std::cout << "\n";
	}*/



	// base non-CUDA implementations
	std::cout << "CPU implementation of NTT\n";
	
	
	std::cout << "    with bit-reverse: ";
	float time = 0;
	for(int i = 0; i < 10; i++) {
		start = clock();
		outVec = inPlaceNTT_DIT(vec, n, p, r, 1);
		end = clock();
		time += (float)(end-start)/CLOCKS_PER_SEC;
	}
	std::cout << time/10.0 << " s\n";
	
	std::cout << "    without bit-reverse: ";
	time = 0;
	for(int i = 0; i < 10; i++) {
		start = clock();
        	outVec = inPlaceNTT_DIT(vec, n, p, r, 0);
        	end = clock();
        	time += (float)(end-start)/CLOCKS_PER_SEC;
	}
	std::cout << time/10.0 << " s\n";


	// profiling CUDA implementations
	std::cout << "\nGPU implementation of NTT\n";


	std::cout << "    with bit-reverse: ";
	time = 0;
	for(int i = 0; i < 10; i++) {
		start = clock();
		outVecGPU = inPlaceNTT_DIT_parallel(vec, n, p, r, 1, nullptr);
		end = clock();
		time += (float)(end-start)/CLOCKS_PER_SEC;
	}
	std::cout <<  time/10.0 << " s\n";
	
	
	std::cout << "    without bit-reverse: ";        
	time = 0;
	for(int i = 0; i < 10; i++) {
		start = clock();
        	outVecGPU = inPlaceNTT_DIT_parallel(vec, n, p, r, 0, nullptr );
        	end = clock();
		time += (float)(end-start)/CLOCKS_PER_SEC;
	}
	std::cout << time/10.0 << " s\n";

	std::cout << "    with bit-reverse and offline modExp: ";
        time = 0;
        for(int i = 0; i < 10; i++) {
                start = clock();
                outVecGPU1 = inPlaceNTT_DIT_parallel(vec, n, p, r, 1, table);
                end = clock();
                time += (float)(end-start)/CLOCKS_PER_SEC;
        }
        std::cout << time/10.0 << " s\n";


	std::cout << "    without bit-reverse and offline modExp: ";
	time = 0; 
	for(int i = 0; i < 10; i++) {
		start = clock();
        	outVecGPU1 = inPlaceNTT_DIT_parallel(vec, n, p, r, 0, table);
        	end = clock();
		time += (float)(end-start)/CLOCKS_PER_SEC;
	}
	std::cout << time/10.0 << " s\n\n";

	std::cout << "Batching:\n";
/*
	int batch_sizes[] = {1, 16, 64, 256, 512, 1024};
	for(int batch_size: batch_sizes) {
		uint64_t **vecs;
		vecs = (uint64_t **)malloc(batch_size*sizeof(uint64_t *));
	
		uint64_t **vecs_res;
		vecs_res = (uint64_t **)malloc(batch_size*sizeof(uint64_t *));

		for(int i = 0; i < batch_size; i++)
			*(vecs + i) = randVec(n);
	
		std::cout << "    online and no bit reversal for batch size " << batch_size << ": ";

		start = clock();
       		vecs_res = inPlaceNTT_DIT_parallel_batched(batch_size, vecs, n, p, r, 0, nullptr);
       		end = clock();
		std::cout <<  (float)(end-start)/CLOCKS_PER_SEC << " s\n";	
	}
*/
	int batch_sizes[] = {1, 16, 64, 256, 512, 1024};
        for(int batch_size: batch_sizes) {
                uint64_t **vecs;
                vecs = (uint64_t **)malloc(batch_size*sizeof(uint64_t *));

                uint64_t **vecs_res;
                vecs_res = (uint64_t **)malloc(batch_size*sizeof(uint64_t *));

                for(int i = 0; i < batch_size; i++)
                        *(vecs + i) = randVec(n);

                std::cout << "    online and bit reversal for batch size " << batch_size << ": ";

                start = clock();
                vecs_res = inPlaceNTT_DIT_parallel_batched(batch_size, vecs, n, p, r, 1, nullptr);
                end = clock();
                std::cout <<  (float)(end-start)/CLOCKS_PER_SEC << " s\n";
        }

	bool res = compVec(outVecGPU, outVec, n, 0);
	std::cout << "\n---> Implementations match?: ";
	if(res) { std::cout << "yes\n"; }
	else { std::cout << "no\n"; }


	return 0;
}
