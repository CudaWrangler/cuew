// cuewTest.cpp : Defines the entry point for the console application.
//

#include <stdlib.h>
#include <stdio.h>
#include "cuew.h"

int main(int argc, char* argv[]) {
  (void) argc;  // Ignored.
  (void) argv;  // Ignored.

  if (cuewInit(CUEW_INIT_CUDA) == CUEW_SUCCESS) {
    printf("CUDA found\n");
    printf("NVCC path: %s\n", cuewCompilerPath());
    printf("NVCC version: %d\n", cuewCompilerVersion());
  }
  else {
    printf("CUDA not found\n");
  }

  if (cuewInit(CUEW_INIT_NVRTC) == CUEW_SUCCESS) {
    int major, minor;
    nvrtcVersion(&major, &minor);
    printf("NVRTC found\n");
    printf("Found runtime compilation library version %d.%d\n",
           major,
           minor);
  }
  else {
    printf("NVRTC not found\n");
  }

  return EXIT_SUCCESS;
}
