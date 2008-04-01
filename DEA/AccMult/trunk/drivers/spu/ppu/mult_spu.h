#ifndef __MULT_SPU_H__
#define __MULT_SPU_H__

#define _GNU_SOURCE
#include <sched.h>

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <common/threads.h>
#include <common/util.h>
#include <core/jobs.h>
#include <common/parameters.h>

#include <datawizard/copy-driver.h>

#include <libspe2.h>

#ifndef MAXSPUS
#define MAXSPUS	4
#endif

#define OK              0
#define TRYAGAIN        1
#define FATAL           2

typedef struct spu_worker_arg_t {
	unsigned deviceid;
	spe_context_ptr_t speid;
	int bindid;
	volatile uint32_t ready_flag  __attribute__ ((aligned(16)));
	unsigned memory_node;
} spu_worker_arg;

void *spu_worker(void *);

unsigned get_spu_count(void);

#endif //  __MULT_SPU_H__
