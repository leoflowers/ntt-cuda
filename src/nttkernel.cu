#include <cmath>		/* log2(), pow() */
#include <cstdint>		/* uint64_t */
#include <cstdlib> 		/* malloc() */
#include <iostream>
#include <cuda.h>

#include "../include/utils.h"	/* bit_reverse(), modExp(), modulo() */
#include "../include/utils.cuh"

__global__
void kernel_loop_body(uint64_t p, uint64_t n, uint64_t a, uint64_t m, uint64_t *d_vec, uint64_t *d_result, bool single_thread) {
	//uint64_t width = (int)(n/m);
	uint64_t l = threadIdx.x;
	uint64_t j;

	if(single_thread) { 
		for(j = 0; j < n; j+=m) {
			for(uint64_t k = 0; k < m/2; k++) {
				uint64_t j1 = j+k;
				uint64_t j2 = j1 + (m/2);

				if(j1 < n && j2 < n) {
					uint64_t factor1 = d_result[j1];
					uint64_t factor2 = modulo_k(modExp_k(a, k, p) * d_result[j2], p);
			
					d_result[j1] 		= modulo_k(factor1 + factor2, p);
					d_result[j2] 		= modulo_k(factor1 - factor2, p);
				}
			}
		}
	}
	else {
		j = m*l;
		for(uint64_t k = 0; k < m/2; k++) {
			uint64_t j1 = j+k;
			uint64_t j2 = j1 + (m/2);

			if(j1 < n && j2 < n) {
				uint64_t factor1 = d_result[j1];
				uint64_t factor2 = modulo_k(modExp_k(a, k, p) * d_result[j2], p);
			
				d_result[j1] 		= modulo_k(factor1 + factor2, p);
				d_result[j2] 		= modulo_k(factor1 - factor2, p);
			}
		}
	}	
}


__global__
void inPlaceNTT_kernel(uint64_t p, uint64_t n, uint64_t r, uint64_t *d_vec, uint64_t *d_result) {
	
	uint64_t i;
	if(threadIdx.x < 13)
		i = threadIdx.x + 1; 							// this will be given to thread according to block index
	else
		return;
	
	uint64_t m = powf(2.0, (float)i);					// calculated within kernel
	uint64_t k_ = (p - 1)/m;							// we need to pass p into kernel
	uint64_t a = modExp_k(r, k_, p);					// calculated within kernel

	__syncthreads();
	if(i == 13)
		cudaDeviceSynchronize();
	else if(i < 4)
		kernel_loop_body<<<1, 1>>>(p, n, a, m, d_vec, d_result, true);
	else if(i < 13 && i >= 4)
		kernel_loop_body<<<1, (n/m)>>>(p, n, a, m, d_vec, d_result, false);
	__syncthreads();
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
__host__
uint64_t *inPlaceNTT_DIT_parallel(uint64_t *h_vec, uint64_t n, uint64_t p, uint64_t r, bool rev){
	uint64_t *h_result;
	cudaError_t err;

	h_result = (uint64_t *) malloc(n*sizeof(uint64_t));

	if(rev) {
		h_result = bit_reverse(h_vec, n);
	} else {
		for(uint64_t i = 0; i < n; i++) {
			h_result[i] = h_vec[i];
		}
	}

	/* kernel stuff */ 
	
	// setting up vector for kernel
	uint64_t size = n * sizeof(uint64_t);

    uint64_t *d_vec;
	err = cudaMalloc((void **) &d_vec, size);
	if(err != cudaSuccess) { 
		std::cout << "cuda error: something went wrong allocating on device\n";
	}

	err = cudaMemcpy(d_vec, h_vec, size, cudaMemcpyHostToDevice);
	if(err != cudaSuccess) { 
		std::cout << "cuda error: something went wrong copying to device\n";
	}

    // setting up results vector
    uint64_t *d_result;
    cudaMalloc((void **) &d_result, size);
    cudaMemcpy(d_result, h_result, size, cudaMemcpyHostToDevice);


	inPlaceNTT_kernel<<<1, 13>>>(p, n, r, d_vec, d_result);
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess) { 
		std::cout << "cuda error: something went wrong with threads\n";
	}

	cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
	cudaFree(d_vec);
	cudaFree(d_result);
	/* end of kernel stuff */
	

	return h_result;
}


