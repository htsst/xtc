set (
  XTC_CUDA_HEADERS
  ${CMAKE_CURRENT_LIST_DIR}/error.h
  ${CMAKE_CURRENT_LIST_DIR}/misc.h
  ${CMAKE_CURRENT_LIST_DIR}/timer.h
  )

install (
  FILES
  ${XTC_CUDA_HEADERS}
  DESTINATION
  include/xtc/cuda
  )
