find_package (MPI)
if (MPI_FOUND)
  include ("${CMAKE_CURRENT_LIST_DIR}/xtc/mpi/CMakeLists.txt")

  set(XTC_HEADERS ${XTC_HEADERS} "${CMAKE_CURRENT_LIST_DIR}/xtc/mpi.h")
endif (MPI_FOUND) 

find_package (CUDA)
if (CUDA_FOUND)
  include ("${CMAKE_CURRENT_LIST_DIR}/xtc/cuda/CMakeLists.txt")
  set(XTC_HEADERS ${XTC_HEADERS} "${CMAKE_CURRENT_LIST_DIR}/xtc/cuda.h")
endif (CUDA_FOUND)

find_package (OpenMP)
if (OPENMP_FOUND)
  set(XTC_HEADERS ${XTC_HEADERS} "${CMAKE_CURRENT_LIST_DIR}/xtc/omp.h")
endif (OPENMP_FOUND)

include ("${CMAKE_CURRENT_LIST_DIR}/xtc/cpp/CMakeLists.txt")

install (
  FILES
  ${XTC_HEADERS}
  DESTINATION
  include/xtc
  )
