#ifndef __WORKERS_H__
#define __WORKERS_H__

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
#include <drivers/cuda/driver_cuda.h>
#endif

#ifdef USE_CUBLAS
#include <drivers/cublas/driver_cublas.h>
#endif

#ifdef USE_SPU
#include <drivers/spu/ppu/driver_spu.h>
#endif

#ifdef USE_GORDON
#include <drivers/gordon/driver_gordon.h>
#endif

//#ifdef USE_CPUS
#include <drivers/core/driver_core.h>
//#endif

#ifndef COMPARE_SEQ
//#define COMPARE_SEQ   1
#endif

#include <datawizard/coherency.h>
#include <datawizard/copy-driver.h>

void init_machine(void);
void init_workers(void);
void terminate_workers(void);
void kill_all_workers(void);
void display_general_stats(void);

void push_codelet_output(buffer_descr *descrs, unsigned nbuffers, uint32_t mask);
void fetch_codelet_input(buffer_descr *descrs, unsigned nbuffers);

#endif // __WORKERS_H__
