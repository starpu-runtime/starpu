/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#ifndef __CG_H__
#define __CG_H__

#include <starpu.h>
#include <common/config.h>

/* we do not necessarily want to allocate room for 256 dependencies, but we
   want to handle the few situation where there are a lot of dependencies as
   well */
#define DYNAMIC_DEPS_SIZE	1

/* randomly choosen ! */
#ifndef DYNAMIC_DEPS_SIZE
#define NMAXDEPS	256
#endif

/* Completion Group list */
struct cg_list_s {
	unsigned nsuccs; /* how many successors ? */
	unsigned ndeps; /* how many deps ? */
	unsigned ndeps_completed; /* how many deps are done ? */
#ifdef DYNAMIC_DEPS_SIZE
	unsigned succ_list_size;
	struct cg_s **succ;
#else
	struct cg_s *succ[NMAXDEPS];
#endif
};

/* Completion Group */
typedef struct cg_s {
	unsigned ntags; /* number of tags depended on */
	unsigned remaining; /* number of remaining tags */
	struct tag_s *tag; /* which tags depends on that cg ?  */

	unsigned completed;

	/* in case this completion group is related to an application, we have
 	 * to explicitely wake the waiting thread instead of reschedule the
	 * corresponding task */
	unsigned used_by_apps;
	pthread_mutex_t cg_mutex;
	pthread_cond_t cg_cond;
} cg_t;

#endif // __CG_H__
