// cuewTest.cpp : Defines the entry point for the console application.
//

#include <stdlib.h>
#include <stdio.h>
#include "cuew.h"

int main(int argc, char* argv[]) {
  if (cuewInit() == CUEW_SUCCESS) {
    printf("CUDA found\n");
    printf("NVCC path: %s\n", cuewCompilerPath());
    printf("NVCC version: %d\n", cuewCompilerVersion());
    if (nvrtcVersion) {
        int major, minor;
        nvrtcVersion(&major, &minor);
        printf("Found runtime compilation library version %d.%d\n",
               major,
               minor);
    }
    else {
        printf("Runtime compilation library is missing\n");
    }
  }
  else {
    printf("CUDA not found\n");
  }
  return EXIT_SUCCESS;
}
