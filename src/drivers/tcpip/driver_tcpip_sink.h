/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2025  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DRIVER_TCPIP_SINK_H__
#define __DRIVER_TCPIP_SINK_H__

/** @file */

#include <drivers/mp_common/sink_common.h>

#pragma GCC visibility push(hidden)

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE

void _starpu_tcpip_sink_init(struct _starpu_mp_node *node);
void _starpu_tcpip_sink_bind_thread(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED, int coreid, int * core_table, int nb_core);

#endif  /* STARPU_USE_TCPIP_MASTER_SLAVE */

#pragma GCC visibility pop

#endif	/* __DRIVER_TCPIP_SINK_H__ */
