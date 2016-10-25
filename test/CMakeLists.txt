project (occ-test)
cmake_minimum_required (VERSION 2.8)

find_package (MPI)

include_directories (
  "${OCC_INSTALL_PERFIX}"
  "${MPI_CXX_INCLUDE_PATH}"
  )

Message("${OCC_INSTALL_PERFIX}")

add_executable (
  occ-test
  test.cc
  )

target_link_libraries (
  occ-test
  ${MPI_CXX_LIBRARIES}
  )
