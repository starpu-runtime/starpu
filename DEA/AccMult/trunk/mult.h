#ifndef __MULT_H__
#define __MULT_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include "util.h"
#include "jobs.h"
#include "parameters.h"

#ifdef USE_CUDA
#include "mult_cuda.h"
#endif

#ifdef USE_CELL
#error not supported yet
#include "mult_cell.h"
#endif

typedef struct core_worker_arg_t {
	int coreid;
} core_worker_arg;


#endif // __MULT_H__
