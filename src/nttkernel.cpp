#include <cmath>		/* log2(), pow() */
#include <cstdint>		/* uint64_t */
#include <cstdlib> 		/* malloc() */
#include <iostream>

#include "../../include/utils.h"	/* bit_reverse(), modExp(), modulo() */
#include "../../include/ntt.h" 	//INCLUDE HEADER FILE

#include <cuda.h>


__global__ void inPlaceNTT_subkernel(


__global__ void inPlaceNTT_kernel(uint64_t p, uint64_t n, uint64_t *d_vec, uint64_t *d_result) {
	uint64_t i = blockIdx.x + 1; 			// this will be given to thread according to block index
	uint64_t m = pow(2,i);				// calculated within kernel
	uint64_t k_ = (p - 1)/m;			// we need to pass p into kernel
	uint64_t a = modExp(r, k_, p);			// calculated within kernel

	
	dim3 
	dim3
	dim3
	inPlaceNTT_subkernel<<< >>>();


	for(uint64_t j = 0; j < n; j += m) {
		for(uint64_t k = 0; k < m/2; k++) { 
			factor1 = d_result[j + k];
			factor2 = modulo(modExp(a, k, p) * d_result[j + k + m/2], p);

			d_result[j + k] 		= modulo(factor1 + factor2, p);
			d_result[j + k+m/2] 		= modulo(factor1 - factor2, p);
		}
	}
}



/**
 * Perform an in-place iterative breadth-first decimation-in-time Cooley-Tukey NTT on an input vector and return the result
 *
 * @param vec 	The input vector to be transformed
 * @param n	The size of the input vector
 * @param p	The prime to be used as the modulus of the transformation
 * @param r	The primitive root of the prime
 * @param rev	Whether to perform bit reversal on the input vector
 * @return 	The transformed vector
 */
uint64_t *inPlaceNTT_DIT_parallel(uint64_t *h_vec, uint64_t n, uint64_t p, uint64_t r, bool rev){
	uint64_t *h_result;
	uint64_t m, k_, a, factor1, factor2;
	result = (uint64_t *) malloc(n*sizeof(uint64_t));


	if(rev) {
		result = bit_reverse(vec, n);
	} else {
		for(uint64_t i = 0; i < n; i++) {
			result[i] = vec[i];
		}
	}


	/* kernel stuff */ 
	// bulk of parallelization
	dim3 dim_grid(log2(n), 1, 1);		// blocks for each of the first for loop iterations
	dim3 dim_block(1, 1, 1);		// starting with one thread per block

	// setting up vector for kernel
        uint64_t *d_vec;
        cudaMalloc((void **) &d_vec, n);
        cudaMemcpy(d_vec, h_vec, cudaMemcpyHostToDevice);

        // setting up results vector
        uint64_t *d_result;
        cudaMalloc((void **) &d_result, n);
        cudaMemcpy(d_result, h_reslt, cudaMemcpyHostToDevice);


	inPlaceNTT_kernel<<<dim_grid, dim_block>>>(p, n, d_vec, d_result);


	cudaMemcpy(h_result, d_result, n, cudaMemcpyDeviceToHost);

	cudaFree(d_vec);
	cudaFree(d_result)
	/* end of kernel stuff */
	

	return h_result;
}

