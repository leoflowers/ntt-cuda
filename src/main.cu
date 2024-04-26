#include <cmath>
#include <cstdint>
#include <ctime>
#include <iostream>
#include <unistd.h>
#include <iomanip>

#include "../include/ntt.h"
#include "../include/utils.h"
#include "../include/nttkernel.cuh"

#define ITERS 100

int main(int argc, char *argv[]) {
	uint64_t *outVec, *outVecGPU, *outVecGPU1;
	uint64_t n = 4096;
	uint64_t p = 68719403009;
	uint64_t r = 36048964756;
	int c = log2(n);

	uint64_t vec[n];
  	for (int i = 0; i < n; i++)
    	vec[i] = i;

	const auto *table = offlineModExp(r, p, n);

	float time = 0;
	clock_t start, end;

	// base non-CUDA implementations
	std::cout << "CPU implementation of NTT\n";
	std::cout << "    with bit-reverse: ";
	for (int i = 0; i < 10; i++) {
		start 	= clock();
		outVec	= inPlaceNTT_DIT(vec, n, p, r, 1);
		end 	= clock();
		time 	+= static_cast<float>(end - start) / CLOCKS_PER_SEC;
	}
	std::cout << time / 10.0 << " s\n";
	
	std::cout << "    without bit-reverse: ";
	time = 0;
	for (int i = 0; i < 10; i++) {
		start 	= clock();
        outVec 	= inPlaceNTT_DIT(vec, n, p, r, 0);
        end 	= clock();
        time 	+= static_cast<float>(end - start) / CLOCKS_PER_SEC;
	}
	std::cout << time / 10.0 << " s\n";

	// profiling CUDA implementations
	std::cout << "\nGPU implementation of NTT\n";
	std::cout << "    with bit-reverse: ";
	time = 0;
	for (int i = 0; i < 10; i++) {
		start 		= clock();
		outVecGPU 	= inPlaceNTT_DIT_parallel(vec, n, p, r, 1, nullptr);
		end 		= clock();
		time 		+= static_cast<float>(end - start) / CLOCKS_PER_SEC;
	}
	std::cout << time / 10.0 << " s\n";
	
	std::cout << "    without bit-reverse: ";        
	time = 0;
	for (int i = 0; i < 10; i++) {
		start 		= clock();
        outVecGPU 	= inPlaceNTT_DIT_parallel(vec, n, p, r, 0, nullptr);
        end 		= clock();
		time 		+= static_cast<float>(end - start) / CLOCKS_PER_SEC;
	}
	std::cout << time / 10.0 << " s\n";

	std::cout << "    with bit-reverse and offline modExp: ";
    time = 0;
    for (int i = 0; i < 10; i++) {
        start 		= clock();
        outVecGPU1 	= inPlaceNTT_DIT_parallel(vec, n, p, r, 1, table);
        end 		= clock();
        time 		+= static_cast<float>(end - start) / CLOCKS_PER_SEC;
    }
    std::cout << time / 10.0 << " s\n";

	std::cout << "    without bit-reverse and offline modExp: ";
	time = 0; 
	for (int i = 0; i < 10; i++) {
		start 		= clock();
        outVecGPU1 	= inPlaceNTT_DIT_parallel(vec, n, p, r, 0, table);
        end 		= clock();
		time 		+= static_cast<float>(end - start) / CLOCKS_PER_SEC;
	}
	std::cout << time / 10.0 << " s\n\n";

	std::cout << "Batching:\n";
	int batch_sizes[] = {1, 16}; //, 64, 256, 512, 1024};
    for (const auto size : batch_sizes) {
        uint64_t **vecs;
        vecs = (uint64_t **)malloc(size*sizeof(uint64_t *));

        uint64_t **vecs_res;
        vecs_res = (uint64_t **)malloc(size*sizeof(uint64_t *));

        for (int i = 0; i < size; i++)
        	*(vecs + i) = randVec(n);

        std::cout << "    online and bit reversal for batch size " << size << ": ";

        start = clock();
        vecs_res = inPlaceNTT_DIT_parallel_batched(size, vecs, n, p, r, 1, nullptr);
        end = clock();
        std::cout << static_cast<float>(end - start) / CLOCKS_PER_SEC << " s\n";
    }

	const auto res = compVec(outVecGPU, outVec, n, 0);
	std::cout << "\n---> Implementations match?: "
		<< (res ? "yes" : "no") << "\n";

	return 0;
}
