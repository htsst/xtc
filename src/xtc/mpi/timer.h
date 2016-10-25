#ifndef XTC_MPI_TIMER_H_
#define XTC_MPI_TIMER_H_

namespace xtc {
namespace mpi {

class Timer {
 public:

  Timer() {
    start_ = 0.0;
    stop_ = 0.0;
  }

  void Start() {
    start_ = MPI::Wtime();
  }

  void StartWithBarrier() {
    communicator().Barrier();
    Start();
  }

  double Stop() {
    stop_ = MPI::Wtime();
    return (stop_ - start_) * 1e+3;
  }
  
  double StopWithBarrier() {
    communicator().Barrier();
    return Stop();
  }
  
 private:
  double start_, stop_;
};

}} // namespace xtc::mpi

#endif // XTC_MPI_TIMER_H_
