// cuewTest.cpp : Defines the entry point for the console application.
//

#include <stdlib.h>
#include <stdio.h>
#include "cuew.h"

int main(int argc, char* argv[]) {
  if (cuewInit()) {
    printf("CUDA found\n");
    printf("NVCC path: %s\n", cuewCompilerPath());
    printf("NVCC version: %d\n", cuewCompilerVersion());
  }
  else {
    printf("CUDA not found\n");
  }
  return EXIT_SUCCESS;
}
