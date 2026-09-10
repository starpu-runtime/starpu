/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2024-2026  University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __ENERGY_COUNTERS_H__
#define __ENERGY_COUNTERS_H__

#include <profiling/fxt/fxt.h>
#include <starpu_data_interfaces.h>
#include <profiling/starpu_tracing.h>
#include <time.h>
#include <limits.h>
#include <core/workers.h>

#ifdef STARPU_HAVE_ENERGYREADER
#include <energy_reader.h>
#endif

/**
 * @file
 * @brief Hardware energy counters monitoring, through the energy-reader
 * library.
 */

/**
 * @brief Probe the available energy-reader backends, register the
 * detected topology (packages, cores, GPUs) and the StarPU workers bindings,
 * before starting the hardware energy counters sampling if the
 * environment variable STARPU_ENERGY_READER is set to 1.
 *
 * @param starpu_config_arg The machine configuration being initialized,
 * used to enumerate the workers to register.
 */
void _starpu_energyreader_init(struct _starpu_machine_config starpu_config_arg);

/**
 * @brief Print a CSV summary of the hardware energy counters, if energy
 * monitoring was started by _starpu_energyreader_init(), and release the
 * resources allocated for it.
 *
 * Called once from starpu_shutdown().
 */
void _starpu_energyreader_terminate(void);

/**
 * @brief Take a new sample of each family of hardware energy counters
 * (CPU package/DRAM, Cray, GPU) whose configured sampling interval has
 * elapsed since the previous sample, and record it into the FxT trace.
 *
 * This is called from the CPU workers' driver loop.
 *
 * @param workerid Id of the CPU worker triggering the measurement, only
 * used to annotate the resulting trace events.
 * @return 0 on success, -1 if a counter could not be read.
 */
int _starpu_energyreader_try_measurement(int workerid);

#endif /* __ENERGY_COUNTERS_H__ */
