#ifndef __MULT_H__
#define __MULT_H__

#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <unistd.h>
#include <string.h>
#include <common/threads.h>
#include <common/util.h>
#include "jobs.h"
#include <common/parameters.h>
//#include "comp.h"

#ifdef USE_CUDA
#include <drivers/cuda/mult_cuda.h>
#endif

#ifdef USE_CUBLAS
#include <drivers/cublas/mult_cublas.h>
#endif

#ifdef USE_CELL
#error not supported yet
#include <drivers/cell/mult_cell.h>
#endif

//#ifdef USE_CPUS
#include <drivers/core/mult_core.h>
//#endif

#ifndef COMPARE_SEQ
//#define COMPARE_SEQ   1
#endif

#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>

void init_machine(void);
void init_workers(void);
void terminate_workers(void);
void display_stats(job_descr *);
void kill_all_workers(void);
void display_general_stats(void);

#endif // __MULT_H__
