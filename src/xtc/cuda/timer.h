#ifndef XTC_CUDA_TIMER_H_
#define XTC_CUDA_TIMER_H_

namespace xtc {
namespace cuda {

class Timer {
 public:

  Timer() {
    cudaEventCreate(&start_);
    cudaEventCreate(&stop_);
  }

  void Start() {
    cudaEventRecord(start_, 0);
  }

  float Stop() {
    cudaEventRecord(stop_, 0);
    cudaEventSynchronize(stop_);
    return elapsed_time();
  }

  float elapsed_time() {
    float elapsed_time_ms = 0.0;
    cudaEventElapsedTime(&elapsed_time_ms, start_, stop_);
    return elapsed_time_ms;
  }

 private:
  cudaEvent_t start_, stop_;
  
};

}} // namespace xtc::cuda

#endif // XTC_CUDA_TIMER_H_
