/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
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

#ifndef __DRIVER_CPU_H__
#define __DRIVER_CPU_H__

#include <common/config.h>
#include <core/jobs.h>

#include <core/perfmodel/perfmodel.h>
#include <common/fxt.h>
#include <datawizard/datawizard.h>

#include <starpu.h>

void *_starpu_cpu_worker(void *);

#ifndef STARPU_NMAXCPUS
#define STARPU_NMAXCPUS       4
#endif

#endif //  __DRIVER_CPU_H__
