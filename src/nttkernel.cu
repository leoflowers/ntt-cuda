#include <cmath>		/* log2(), pow() */
#include <cstdint>		/* uint64_t */
#include <cstdlib> 		/* malloc() */
#include <iostream>
#include <cuda.h>

#include "../include/utils.h"	/* bit_reverse(), modExp(), modulo() */
#include "../include/utils.cuh"

__global__
void kernel_loop_body(int c, uint64_t p, uint64_t n, uint64_t a, uint64_t m, uint64_t *d_vec, uint64_t *d_result, bool single_thread, const uint64_t *table) {
	
	uint64_t l = threadIdx.x;
	uint64_t j;
	uint64_t y = c/2;

	if(single_thread) {							 
		for(j = 0; j < n; j+=m) {
			for(uint64_t k = 0; k < m/2; k++) {
				uint64_t j1 = j+k;
				uint64_t j2 = j1 + (m/2);

				if(j1 < n && j2 < n) {
					uint64_t factor1 = d_result[j1];
					
					// sets up arg depending on online/offline
					uint64_t arg;
					if(table == nullptr) 
						arg = modExp_k(a, k, p); 
					else
						arg = *(table + (c*y)  + (k+1));
					uint64_t factor2 = modulo_k(arg * d_result[j2], p);
			
					d_result[j1] 		= modulo_k(factor1 + factor2, p);
					d_result[j2] 		= modulo_k(factor1 - factor2, p);
				}
			}
		}
	}
	else {
		j = m*l;				// more parallelism to be extracted from here
		for(uint64_t k = 0; k < m/2; k++) {
			uint64_t j1 = j+k;
			uint64_t j2 = j1 + (m/2);

			if(j1 < n && j2 < n) {
				uint64_t factor1 = d_result[j1];
				
				// sets up arg depending on online/offline
                                uint64_t arg;
                                if(table == nullptr)
                                	arg = modExp_k(a, k, p);
                                else
                                        arg = *(table + (c*y)  + (k+1));
				uint64_t factor2 = modulo_k(arg * d_result[j2], p);
		
				d_result[j1] 		= modulo_k(factor1 + factor2, p);
				d_result[j2] 		= modulo_k(factor1 - factor2, p);
			}
		}
	}
}


__global__
void inPlaceNTT_kernel(int c, uint64_t p, uint64_t n, uint64_t r, uint64_t *d_vec, uint64_t *d_result, const uint64_t *table) {
	
	uint64_t i;
	if(threadIdx.x < 13)
		i = threadIdx.x + 1;			// this will be given to thread according to block index
	else
		return;
	

	uint64_t m = powf(2.0, (float)i);		// calculated within kernel
	uint64_t k_ = (p - 1)/m;
	uint64_t a;
	if(table != nullptr) 	
		a = *(table + (i-1)*c);
	else 
		a = modExp_k(r, k_, p);	


	// spawns threads as necessary
	__syncthreads();
	if(i == 13) 
		cudaDeviceSynchronize();
	else if(i < 4) 
		kernel_loop_body<<<1, 1>>>(c, p, n, a, m, d_vec, d_result, true, table); 
	else if(i < 13 && i >= 4)
		kernel_loop_body<<<1, (n/m)>>>(c, p, n, a, m, d_vec, d_result, false, table);
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
uint64_t *inPlaceNTT_DIT_parallel(uint64_t *h_vec, uint64_t n, uint64_t p, uint64_t r, bool rev, const uint64_t *table){
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
	int i_ = log2(n);

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

	uint64_t *d_table;
	if(table != nullptr) {
		uint64_t table_size = (i_ * i_)*sizeof(uint64_t);
		cudaMalloc((void **) &d_table, table_size);
		cudaMemcpy(d_table, table, table_size, cudaMemcpyHostToDevice);
	}
	else {
		d_table = nullptr;
	}


	dim3 dim_block(i_ + 1, 1, 1);
	inPlaceNTT_kernel<<<1, dim_block>>>(i_, p, n, r, d_vec, d_result, d_table);
	err = cudaDeviceSynchronize();
	if(err != cudaSuccess) { 
		std::cout << "cuda error: something went wrong with threads\n";
	}


	cudaMemcpy(h_result, d_result, size, cudaMemcpyDeviceToHost);
	cudaFree(d_vec);
	cudaFree(d_result);
	if(table != nullptr)
		cudaFree(d_table);
	/* end of kernel stuff */
	

	return h_result;
}

__host__
uint64_t **inPlaceNTT_DIT_parallel_batched(int batch_size, uint64_t **h_vecs, uint64_t n, uint64_t p, uint64_t r, bool rev, const uint64_t *table) {
	uint64_t **results;
	results = (uint64_t **)malloc(batch_size * sizeof(uint64_t *));
	
	for(int i = 0; i < batch_size; i++) {
		*(results+i) = inPlaceNTT_DIT_parallel(*(h_vecs+i), n, p, r, rev, table);
	}

	return results;
}
