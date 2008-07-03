#ifndef __DRIVER_CORE_H__
#define __DRIVER_CORE_H__

/* to bind threads onto a given cpu */
#define _GNU_SOURCE
#include <sched.h>

#include <common/util.h>
#include <common/parameters.h>
#include <core/jobs.h>
#include <core/workers.h>

#include <common/fxt.h>

#include <datawizard/copy-driver.h>

typedef struct core_worker_arg_t {
        int coreid;
        volatile int ready_flag;
	int bindid;
	unsigned memory_node;
} core_worker_arg;

void *core_worker(void *);

#ifndef NMAXCORES
#define NMAXCORES       4
#endif

#endif //  __DRIVER_CORE_H__
