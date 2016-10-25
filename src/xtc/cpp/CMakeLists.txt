set (
  XTC_CPP_HEADERS
  ${CMAKE_CURRENT_LIST_DIR}/timer.h
  )

install(
  FILES
  ${XTC_CPP_HEADERS}
  DESTINATION
  include/xtc/cpp
  )
