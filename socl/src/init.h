/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 *
 * StarPU is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * StarPU is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include "socl.h"
#include "gc.h"
#include "mem_objects.h"

#ifndef SOCL_INIT_H
#define SOCL_INIT_H

extern int _starpu_init_failed;
extern volatile int _starpu_init;
/**
 * Initialize StarPU
 */

int socl_init_starpu(void);
void soclShutdown(void);

#endif /* SOCL_INIT_H */
