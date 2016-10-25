#ifndef XTC_MPI_RUNTIME_H_
#define XTC_MPI_RUNTIME_H_

namespace MPI {
class Intracomm;
}

namespace xtc {
namespace mpi {

char g_processor_name[MPI_MAX_PROCESSOR_NAME];
double g_start_time = 0.0;

MPI::Intracomm& communicator() { return MPI::COMM_WORLD; }

int rank() { return MPI::COMM_WORLD.Get_rank(); }

int size() { return MPI::COMM_WORLD.Get_size(); }

inline bool is_head_rank() { return (rank() == 0); }

inline bool is_tail_rank() { return (rank() == (size() - 1)); }

const char* hostname() { return g_processor_name; }

double GetMPIElapsedTime() { return MPI::Wtime() - g_start_time; }


void Init() {
  if (!MPI::Is_initialized()) {
    MPI::Init();
    communicator().Set_errhandler(MPI_ERRORS_ARE_FATAL);
    
    int processor_name_length;
    MPI::Get_processor_name(g_processor_name, processor_name_length);
    g_start_time = MPI::Wtime();
  }
}

void Destroy() {
  if (MPI::Is_initialized())
    MPI::Finalize();
}

}} // namespace xtc::mpi

#endif // XTC_MPI_RUNTIME_H_
