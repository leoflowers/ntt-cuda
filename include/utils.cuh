#ifndef UTILS_CUH
#define UTILS_CUH

#include <cstdint> 	/* int64_t, uint64_t */
#include <cstdlib>	/* RAND_MAX */
#include <cuda.h>
/**
 * Perform the operation 'base^exp (mod m)' using the memory-efficient method on GPU
 *
 * @param base	The base of the expression
 * @param exp	The exponent of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
__device__
uint64_t modExp_k(uint64_t base, uint64_t exp, uint64_t m);

/**
 * Perform the operation 'base (mod m)' on GPU
 *
 * @param base	The base of the expression
 * @param m	The modulus of the expression
 * @return 	The result of the expression
 */
__device__
uint64_t modulo_k(int64_t base, int64_t m);
#endif
