#ifndef __MULT_H__
#define __MULT_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <math.h>
#include "threads.h"
#include "util.h"
#include "jobs.h"
#include "parameters.h"
#include "comp.h"

#ifdef USE_CUDA
#include "mult_cuda.h"
#endif

#ifdef USE_CUBLAS
#include "mult_cublas.h"
#endif

#ifdef USE_CELL
#error not supported yet
#include "mult_cell.h"
#endif

//#ifdef USE_CPUS
#include "mult_core.h"
//#endif

#ifndef COMPARE_SEQ
//#define COMPARE_SEQ   1
#endif


#endif // __MULT_H__
