/*
 * StarPU
 * Copyright (C) Université Bordeaux 1, CNRS 2008-2010 (see AUTHORS file)
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

#ifndef __SORT_DATA_HANDLES_H__
#define __SORT_DATA_HANDLES_H__

#include <starpu.h>
#include <common/config.h>
#include <stdlib.h>

#include <stdarg.h>
#include <datawizard/coherency.h>
#include <datawizard/memalloc.h>

void _starpu_sort_task_handles(starpu_buffer_descr descr[], unsigned nbuffers);

#endif // SORT_DATA_HANDLES
