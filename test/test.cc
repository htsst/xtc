#include <iostream>
#include <xtc/mpi.h>

int main() {

  xtc::mpi::Init();

  xtc::mpi::Message("Hello World");

  xtc::mpi::Destroy();

  return 0;
}
