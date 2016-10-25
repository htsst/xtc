#ifndef XTC_CUDA_MISC_H_
#define XTC_CUDA_MISC_H_

#include <stdint.h>

namespace xtc {
namespace cuda {

const int kMaxThreadsPerBlock = 1024;
const int WARP_SIZE = 32;

template<typename T>
struct SharedMemory {
  __device__ inline operator T*() {
    extern __shared__ T __smem[];
    return (T*)__smem;
  }

  __device__ inline operator const T*() const {
    extern __shared__ T __smem[];
    return (T*)__smem;
  }
};

template<>
struct SharedMemory<float> {
  __device__ inline operator float*() {
    extern __shared__ float __smem_f[];
    return (float*)__smem_f;
  }

  __device__ inline operator const float*() const {
    extern __shared__ float __smem_f[];
    return (float*)__smem_f;
  }
};

template<>
struct SharedMemory<double> {
  __device__ inline operator double*() {
    extern __shared__ double __smem_d[];
    return (double*)__smem_d;
  }

  __device__ inline operator const double*() const {
    extern __shared__ double __smem_d[];
    return (double*)__smem_d;
  }
};

template<>
struct SharedMemory<int> {
  __device__ inline operator int*() {
    extern __shared__ int __smem_i[];
    return (int*)__smem_i;
  }

  __device__ inline operator const int*() const {
    extern __shared__ int __smem_i[];
    return (int*)__smem_i;
  }
};

inline uint32_t GetNextPowerOf2(uint32_t x) {
  --x;
  x |= x >> 1;
  x |= x >> 2;
  x |= x >> 4;
  x |= x >> 8;
  x |= x >> 16;
  return ++x; 
}

inline void GetNumBlocksAndNumThreadsDefault(int n,
                                             int *num_blocks_ptr,
                                             int *num_threads_ptr) {
  int num_blocks = 0, num_threads = 0;
  int max_threads_per_block = kMaxThreadsPerBlock;

  if (n < max_threads_per_block) {
    num_blocks = 1;
    num_threads = n;
  } else {
    num_blocks = (n / max_threads_per_block) + 1;
    num_threads = max_threads_per_block;
  }
    
  *num_blocks_ptr = num_blocks;
  *num_threads_ptr = num_threads;
}

inline void GetGridAndBlockBasedOnWarpSizeAndElements(int num_elements,
                                                      dim3 *grid_ptr,
                                                      dim3 *block_ptr) {

  dim3 grid(1), block(1);
  const int max_threads_per_block = kMaxThreadsPerBlock;
  const int warp_size = WARP_SIZE;
  const int max_elements_per_block = max_threads_per_block / warp_size;

  if (num_elements < max_elements_per_block) {
    block.x = num_elements;
    // grid.x = 1;
  } else {
    block.x = max_elements_per_block;
    grid.x = num_elements / max_elements_per_block + 1;
  }

  block.y = warp_size;
  
  *grid_ptr = grid;
  *block_ptr = block;
}

// assign 32 threads (== blockDim.y) per element
inline void GetNumBlocksAndNumThreadsWarpPerElement(int n,
						    dim3 *num_blocks_ptr,
						    dim3 *num_threads_ptr) {
  dim3 num_blocks(1), num_threads(1);
  int max_threads_per_block = kMaxThreadsPerBlock;
  int warp_size = WARP_SIZE;

  // assign 32 threads per element
  num_threads.y = warp_size;

  if (n < max_threads_per_block) {
    num_threads.x = n;
    num_blocks.x = 1;
    printf("UUU %d %d\n", num_threads.x, max_threads_per_block);
  } else {
    //num_threads.x = (min(n, max_threads_per_block) + 1) / warp_size;
    num_threads.x = min((n / warp_size + 1), max_threads_per_block / warp_size);
    num_blocks.x = (n / num_threads.x) + 1;
    printf("XXX %d\n", num_threads.x);
  }

  *num_blocks_ptr = num_blocks;
  *num_threads_ptr = num_threads;
}

inline void PrintMemoryUsageInMebiByte(const char *message) {
#if DISCCL_DEBUG
  size_t free, total;
  CUDASafeCall(cudaMemGetInfo(&free, &total));
  std::cout << message 
            << "Use: " << (total - free) / (1024 * 1024) << " MiB, " 
            << "Free: " << free / (1024 * 1024) << " MiB, "
            << "Total: " << total / (1024 * 1024) << " MiB " << std::endl;
#endif
}

}} // namespace xtc::cuda

#endif // XTC_CUDA_MISC_H_
