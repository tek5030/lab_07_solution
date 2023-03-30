#include "lab_stereo.h"
#include <iostream>

int main() try
{
  lab_stereo();

  return EXIT_SUCCESS;
}
catch (const std::exception& e)
{
  std::cerr << "Caught exception:\n"
            << e.what() << "\n";
  return EXIT_FAILURE;
}
catch (...)
{
  std::cerr << "Caught unknown exception\n";
  return EXIT_FAILURE;
}