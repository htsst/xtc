#ifndef XTC_CPP_TIMER_H_
#define XTC_CPP_TIMER_H_

#include <sys/time.h>
#include <cstddef>

namespace xtc {
namespace cpp {

class Timer {
 public:
  Timer() {
    start_ = 0.0;
    stop_ = 0.0;
  }

  __inline__ void Start() {
    start_ = GetMilliSecond();
  }

  __inline__ double Stop() {
    stop_ = GetMilliSecond();
    return elapsed_time();
  }

  inline double elapsed_time() {
    return stop_ - start_;
  }

 private:

  double GetMilliSecond() {
    struct timeval t;
    gettimeofday(&t, NULL);
    return static_cast<double>(t.tv_sec) * 1e+3 + static_cast<double>(t.tv_usec) * 1e-3;
  }
  
}
  
  double start_, stop_;

};

}} // namespace xtc::cpp

#endif // XTC_CPP_TIMER_H_
