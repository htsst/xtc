include_directories (
  ${MPI_CXX_INCLUDE_PATH}
  )

set (
  XTC_MPI_HEADERS
  ${CMAKE_CURRENT_LIST_DIR}/runtime.h
  ${CMAKE_CURRENT_LIST_DIR}/timer.h
  ${CMAKE_CURRENT_LIST_DIR}/logger.h
  )

install (
  FILES
  ${XTC_MPI_HEADERS}
  DESTINATION
  include/xtc/mpi
  )
