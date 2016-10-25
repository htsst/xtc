#ifndef XTC_MPI_LOGGER_H_
#define XTC_MPI_LOGGER_H_

#include <cstdio>
#include <cstdarg>
#include "runtime.h"

namespace xtc {
namespace mpi {

void PrintLog(FILE * stream,
              int rank,
              const char *hostname,
              double time,
              const char *severity,
              const char *prefix,
              const char *suffix,
              const char *format,
              va_list argp) {
  fprintf(stream, "{rank: %d, hostname: \"%s\", time: %g, severity: \"%s\", %s",
          rank, hostname, time, severity, prefix);
  vfprintf(stream, format, argp);
  fprintf(stream, "%s}\n", suffix);
}

void Message(const char* format, ...) {
  va_list argp;
  va_start(argp, format);
  PrintLog(stdout, rank(), hostname(), GetMPIElapsedTime(),
           "INFO", "message: \"", "\"", format, argp);
  va_end(argp);
}

void MessageOnMaster(const char *format, ...)  {

  if (is_head_rank()) {
    va_list argp;
    va_start(argp, format);
    PrintLog(stdout, rank(), hostname(), GetMPIElapsedTime(),
             "INFO", "message: \"", "\"", format, argp);
    va_end(argp);
  }
  
  communicator().Barrier();
}

void ErrorMessage(const char* format, ...) {
  va_list argp;
  va_start(argp, format);
  PrintLog(stderr, rank(), hostname(), GetMPIElapsedTime(),
           "ERROR", "message: \"", "\"", format, argp);
  va_end(argp);
}

}} // namespace xtc::mpi

#endif // XTC_MPI_LOGGER_H_
