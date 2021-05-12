#Project repository for ECE-GY 9143 semester project. 

This project involved taking a sequential version of the Cooley-Tukey NTT algorithm and using
CUDA to parallelize the code for use with GPUs. 


## Bit reversal removal
Before discussing the exact speedup of removing bit reversal, I must note that I was able to time
these functions using clock() from the ctime library. I wrapped the function inPlaceNTT_DIT_parallel(),
and so these times reflect the overheads that are normally incurred when running code on GPUs. 

Currently, the main.cu file runs some profiling on the CUDA kernels with appropriate testing conditions.
For profiling the removal of bit reversing, I take the average of 10 instances of the CUDA kernel
with and without bit reversing. For the CUDA kernel not running bit reversal, we have an average
running time of 0.621783 seconds, while average running time without bit reversal is 0.601436 seconds. 
This gives us an decrease of ~0.02 seconds with all things remaining equal. Thus, it's evident that
bit reversal might be some unneccessary overhead for computing the NTT of a vector. However, this
is only tested on one single input and so I cannot speculate too much on how this time difference
with manifest itself with other inputs. Regardless, it is clear that it might be much more
efficient for the totality of the system to perhaps leave something like bit reversal in some
other stage of the pipeline for FHE workloads.

## Twiddle factors
Fair warning: it seems that my logic for implementing the twiddle factor optimization is off
somewhere, as I have not been able to match the original output using twiddle factors.
However, the code that I have in place does seem to be quite close, at least computationally,
to a proper implementation. Once again, I took the average running time out of 10 instances
of the kernel running with the twiddle factor optimization. Below are the outputs from
./ntt-kernel:  

    with bit-reverse: 0.621159 s
    without bit-reverse: 0.601025 s
    with bit-reverse and offline modExp: 0.198285 s
    without bit-reverse and offline modExp: 0.197987 s

From this, we can see that making modExp online adds a lot of overhead to our kernel. Taking it offline
gave us an average improvement of ~0.5 seconds, which would probably be magnified when using bigger
vectors, larger primes, etc. 



[TODO]
