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

#define NMAXGORDONSPUS	8

void *gordon_worker(void *);

#endif // __DRIVER_GORDON_H__
