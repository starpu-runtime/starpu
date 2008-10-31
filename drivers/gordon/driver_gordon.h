#ifndef __DRIVER_GORDON_H__
#define __DRIVER_GORDON_H__

#ifndef _GNU_SOURCE
#define _GNU_SOURCE
#endif
#include <sched.h>

#include <assert.h>
#include <stdio.h>
#include <math.h>

#include <common/threads.h>
#include <common/util.h>
#include <core/jobs.h>
#include <common/parameters.h>


#include <stdint.h>
#include <stdlib.h>

#include <libspe2.h>

#define MAXSPUS	8

#define OK	0
#define	TRYAGAIN	1
#define	FATAL	2

typedef struct gordon_job_descr_t {
	
} gordon_job_descr;

typedef struct gordon_worker_arg_t {
	int deviceid;
	int bindid;
	unsigned nspus;
	volatile int ready_flag;
	unsigned memory_nodes[MAXSPUS];
} gordon_worker_arg;

void *gordon_worker(void *);

#endif // __DRIVER_GORDON_H__
