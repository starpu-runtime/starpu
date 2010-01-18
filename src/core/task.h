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

#ifndef __CORE_TASK_H__
#define __CORE_TASK_H__

#include <starpu.h>
#include <common/config.h>

/* In order to implement starpu_wait_all_tasks, we keep track of the number of
 * task currently submitted */
void _starpu_increment_nsubmitted_tasks(void);
void _starpu_decrement_nsubmitted_tasks(void);

#endif // __CORE_TASK_H__
