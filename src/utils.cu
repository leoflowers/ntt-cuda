#include <cstdint> 		/* int64_t, uint64_t */

#include "../include/utils.cuh" 	//INCLUDE HEADER FILE


/**
 * Perform the operation 'base^exp (mod m)' using the memory-efficient method on GPU
 *
 * @param base	The base of the expression
 * @param exp	The exponent of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
__device__
uint64_t modExp_k(uint64_t base, uint64_t exp, uint64_t m){
	uint64_t result = 1;
	
	while(exp > 0) {
		if(exp % 2) 
			result = modulo_k(result*base, m);

		exp = exp >> 1;
		base = modulo_k(base*base, m);
	}

	return result;
}

/**
 * Perform the operation 'base (mod m)' on GPU
 *
 * @param base	The base of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
__device__
uint64_t modulo_k(int64_t base, int64_t m){
	int64_t result = base % m;

	return (result >= 0) ? result : result + m;
}
