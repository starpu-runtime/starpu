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

typedef struct core_worker_arg_t {
	int coreid;
	volatile int ready_flag;
} core_worker_arg;

#ifndef NMAXCORES
#define NMAXCORES       3
#endif

#ifndef COMPARE_SEQ
#define COMPARE_SEQ   1
#endif


#endif // __MULT_H__
