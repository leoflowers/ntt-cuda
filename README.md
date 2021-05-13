# ECE-GY 9143 Project - NTT-CUDA 

This project involved taking a sequential version of the Cooley-Tukey NTT algorithm and using
CUDA to parallelize the code for use with GPUs. 

### How to run:
On Greene, I was able to run my program using the `make` command after requesting a GPU and
loading `cuda/11.1.74`. While the output produces some errors (some variables not used, 
redundant dependencies), `make` and `nvcc` take care of these and allow the binary
`ntt-kernel` to run successfully. 


## Implementation and performance
### Implementation
This semester, we dove into a new direction of computing architectures as a way to circumvent
the end of Dennard scaling. In particular, highly parallel architectures will become more
prevalent as time progresses. For this project, we explore the architectures found in GPUs,
which are meant to be highly parallelizable. We began with a sequential implementation of
the Cooley-Tukey NTT algorithm and attempted to extract parallelism in this algorithm using
CUDA. This endeavor is interesting as we approach a post-quantum computing world, somewhat
forcing us to find ways to make techniques like fully homomorphic encryption viable for
cryptography-dependent workloads. 

When it came to implementing the NTT algorithm in CUDA, the most important thing to do
at the beginning is to visualize the problem. As we can infer from the name, GPUs work
well with graphics. In particular, CUDA's thread structure really does resemble the
structure of data that GPUs work on. Thus, the first problem of implementation is
having a good understanding of the NTT algorithm. 

After understanding the problem through background reading, the parallelization of
the code is straightforward. The first part of parallelization was noticing that
the algorithm first iterates `i = log2(n)` times, where `n` is the size of the input,
each iteration being completely independent of each other. Thus, we can use CUDA
to spawn one thread for each of these iterations. 

The next bit of parallelization comes from understanding what happens at each
iteration. In each iteration, we have `m = n/i` lists (each with `i` elements)
that are each processed individually. Thus, using CUDA's dynamic parallelism,
we are able to spawn these m threads for each of the `i` original threads.

Now, I believe there is one more level of parallelization that is not implemented
in my code yet. Each of these m lists (of `i` elements each) are processed by pairs
of elements: `(1, i/2), (2, i/2 + 1), (3, i/2 + 3),` etc. Now, in a full parallel
implementation, this would mean that each of the m lists will spawn `i/2` threads
that will perform the bulk calculation of the NTT algorithm (calculating factors
using `modExp(), modulo()`, etc.). As far as I know, this is as far as we can break down the
NTT algorithm into parallel components. 

### Performance
After a working implementation, I ran some profiling (using the same techniques as
described in the bit reversal removal section) in order to get a sense of the speedup.
After testing the code on input vectors of size 4096, using the prime and constants
provided to us, the kernel (including any memory transfers and extra actions that
the kernel does) takes, on average, between ~0.5076 seconds (with none of the optimizations)
on an Quadro RTX 8000. However, I believe I also attained average speeds of ~0.6219 seconds
on some of the other GPUs provided by Greene (I did not do my due diligence and checked which
GPU it was exactly, but I speculate a V100). 

Now, this sounds good on its own, but the CPU implementation was able to complete the
operation in ~0.00390 seconds (on average). The way that I reason through this is that
there's a lot of overhead incurred when transferring/communicating with a GPU. However,
I also believe that this overhead is only upfront, and that when working with sufficiently
big vectors (as one would probably have to in an FHE workload), the amortized overhead
will be negligible. 

## Bit reversal removal
Before discussing the exact speedup of removing bit reversal, I must note that I was able to time
these functions using `clock()` from the `ctime` library. I wrapped the function `inPlaceNTT_DIT_parallel()`,
and so these times reflect the overheads that are normally incurred when running code on GPUs. 

Currently, the `main.cu` file runs some profiling on the CUDA kernels with appropriate testing conditions.
For profiling the removal of bit reversing, I take the average of 10 instances of the CUDA kernel
with and without bit reversing. For the CUDA kernel not running bit reversal, we have an average
running time of 0.621783 seconds, while average running time without bit reversal is 0.601436 seconds. 
This gives us an decrease of ~0.02 seconds with all things remaining equal. Thus, it's evident that
bit reversal might be some unneccessary overhead for computing the NTT of a vector. However, this
is only tested on one single input and so I cannot speculate too much on how this time difference
with manifest itself with other inputs. Regardless, it is clear that it might be much more
efficient for the totality of the system to perhaps leave something like bit reversal in some
other stage of the pipeline for FHE workloads.

This implementation was straightforward. The baseline implementation already has a toggle to
perform the bit reversal using a boolean flag. However, I chose to not parallelize this specific
piece of code because it it feels like bit reversal should be a preprocessing activity rather
than being something that should be handed off to the GPU. 

## Twiddle factors
Fair warning: it seems that my logic for implementing the twiddle factor optimization is off
somewhere, as I have not been able to match the original output using twiddle factors.
However, the code that I have in place does seem to be quite close, at least computationally,
to a proper implementation. Once again, I took the average running time out of 10 instances
of the kernel running with the twiddle factor optimization. Below are the outputs from
`./ntt-kernel`:  

    with bit-reverse: 0.621159 s
    without bit-reverse: 0.601025 s
    with bit-reverse and offline modExp: 0.198285 s
    without bit-reverse and offline modExp: 0.197987 s

From this, we can see that making `modExp()` online adds a lot of overhead to our kernel. Taking it offline
gave us an average improvement of ~0.5 seconds, which would probably be magnified when using bigger
vectors, larger primes, etc. 

As for the implementation, I created a function that runs on the CPU that computes a table containing
the necessary `modExp()` computations. In the code, this function is called `offlineModExp()` and returns
a pointer to the computed table. I then modified my functions such that the kernel accepts a pointer
to the table. The kernel will read from the table if it is present, otherwise it will continue
with online computation. 

## Batching
![alt text](https://github.com/leoflowers/ntt-cuda/blob/master/figs/plot.png?raw=true)

Batching came with the expected (yet still surprising) results: times scales linearly as
batch size increases. This is very evident from the plot above, aside from the differences
in slope. This is expected due to the parallel nature of the GPU. However, this result is
surprising as I was expecting the sheer number of threads running on our SM on the GPU
to affect thread switches. However, this did not turn out to be the case (thankfully).
Although, I am sure that for full size workloads, this would not scale as nicely.

Implementation of batching was pretty straightforward. I created another `__host__` kernel
that takes in a double pointer, each of the pointers pointing to each of the input vectors.
This batching kernel then launches the baseline CUDA NTT kernel on each of the input vectors.
