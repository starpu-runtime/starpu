#ifndef __DRIVER_CORE_H__
#define __DRIVER_CORE_H__

/* to bind threads onto a given cpu */
#define _GNU_SOURCE
#include <sched.h>

#include <common/util.h>
#include <common/parameters.h>
#include <core/jobs.h>

#include <core/perfmodel/perfmodel.h>
#include <common/fxt.h>
#include <datawizard/datawizard.h>

void *core_worker(void *);

#ifndef NMAXCORES
#define NMAXCORES       4
#endif

#endif //  __DRIVER_CORE_H__
