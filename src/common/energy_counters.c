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

#include <common/energy_counters.h>

#ifdef STARPU_HAVE_ENERGYREADER

#define DOUBLE_TO_UL_WITH_PRECISION(value, decimal_precision)                   \
	({                                                                      \
		double factor = pow(10.0, decimal_precision);                   \
		double tempValue = (value) * factor;                            \
		(tempValue > ULONG_MAX) ? ULONG_MAX : (unsigned long)tempValue; \
	})

#define TIME_DIFF_MS(start, end) ((end.tv_sec * 1000L + end.tv_nsec / 1000000L) - \
				  (start.tv_sec * 1000L + start.tv_nsec / 1000000L))
#define TIMESPEC_TO_MS(ts) ((ts).tv_sec * 1000.0 + (ts).tv_nsec / 1e6)

#define MAX_INT(a, b) ((a) > (b) ? (a) : (b))
#define GPU_REFRESH_RATE_NUM_TRIES 25
#define GPU_REFRESH_RATE_MULTIPLIER 5

static volatile int energy_reader_enabled = 0;
static energy_reader_context_t context_g = NULL;
static energy_samples_set_s *rapl_set = NULL;
static energy_samples_set_s *cray_set = NULL;
static energy_samples_set_s *gpu_set = NULL;

// interval timings
static int default_sample_interval_rapl_ms = 50; // good for rapl (updated every 1ms)
static int default_sample_interval_cray_ms = 500;
static int min_sample_interval_gpu_ms = 50;
static int backup_sample_interval_gpu_ms = 500;

static int sample_interval_rapl_ms = -1;
static int sample_interval_cray_ms = -1;
static int sample_interval_gpu_ms = -1;

static struct timespec last_reading_time_rapl = {0, 0};
static struct timespec last_reading_time_cray = {0, 0};
static struct timespec last_reading_time_gpu = {0, 0};

static pthread_mutex_t energy_mutex;

/**
 * @brief Checks if the energy reader is enabled with the STARPU_ENERGY_READER environment variable.
 *
 * @return 1 if enabled, 0 otherwise.
 */
static int _starpu_energyreader_is_enabled(void)
{
	return starpu_getenv_number_default("STARPU_ENERGY_READER", 0);
}

/**
 * @brief Initializes the energy clock for measuring energy consumption at fixed intervals.
 */
static void _starpu_energy_clock_init(void)
{
	_starpu_clock_gettime(&last_reading_time_rapl);
	_starpu_clock_gettime(&last_reading_time_gpu);
	_starpu_clock_gettime(&last_reading_time_cray);
}

/**
 * @brief Initializes energy measurement sample intervals .
 */
static int _starpu_initialize_energy_samples(void)
{
	energy_backend_mask bmask = energy_reader_get_available_backends(context_g);

	// for RAPL, refresh period is well known and fixed on all platforms (1ms)
	if (bmask & (BACKEND_MASK_RAPL_PERF | BACKEND_MASK_RAPL_POWERCAP))
	{
		rapl_set = energy_reader_create_sample_set(COUNTER_MASK_ALL,
							   DOMAIN_MASK_CPU_PKG | DOMAIN_MASK_CPU_MEMORY,
							   BACKEND_MASK_RAPL_PERF | BACKEND_MASK_RAPL_POWERCAP,
							   context_g);
		if (!rapl_set)
		{
			_STARPU_DISP("error creating RAPL sample set");
			return -1;
		}
		sample_interval_rapl_ms = starpu_getenv_number_default("STARPU_ENERGY_PKG_INTERVAL", default_sample_interval_rapl_ms);
		_STARPU_DISP("STARPU_ENERGY_PKG_INTERVAL=%d\n", sample_interval_rapl_ms);
	}

	// cray pm counters are updated every 100 ms
	if (bmask & BACKEND_MASK_CRAY_PM)
	{
		cray_set = energy_reader_create_sample_set(COUNTER_MASK_ALL,
							   DOMAIN_MASK_ALL,
							   BACKEND_MASK_CRAY_PM,
							   context_g);
		if (!cray_set)
		{
			_STARPU_DISP("error creating CRAY PM sample set");
			return -1;
		}
		sample_interval_cray_ms = starpu_getenv_number_default("STARPU_ENERGY_CRAY_INTERVAL", default_sample_interval_cray_ms);
		_STARPU_DISP("STARPU_ENERGY_CRAY_INTERVAL=%d\n", sample_interval_cray_ms);
	}

	// for GPU's, refresh period depends on vendors/models
	if (bmask & (BACKEND_MASK_NVML | BACKEND_MASK_ROCMSMI))
	{
		gpu_set = energy_reader_create_sample_set(COUNTER_MASK_ALL,
							  DOMAIN_MASK_ALL,
							  BACKEND_MASK_NVML | BACKEND_MASK_ROCMSMI,
							  context_g);
		if (!gpu_set)
		{
			_STARPU_DISP("error creating GPU sample set");
			return -1;
		}

		// Environment variable override
		int env_val_sample_gpu = starpu_getenv_number("STARPU_ENERGY_GPU_INTERVAL");
		if (env_val_sample_gpu > 0)
		{
			sample_interval_gpu_ms = env_val_sample_gpu;
			_STARPU_DISP("STARPU_ENERGY_GPU_INTERVAL=%d (set by environment variable)\n", sample_interval_gpu_ms);
		}
		else
		{
			double avg_time_us = 0;
			double std = 0;

			if (bmask & BACKEND_MASK_NVML)
			{
				energy_reader_get_update_period(BACKEND_NVML,
								10, &avg_time_us, &std, context_g);
			}
			else
			{
				energy_reader_get_update_period(BACKEND_ROCMSMI,
								10, &avg_time_us, &std, context_g);
			}

			int computed_refresh_rate_ms = (int)((avg_time_us / 1000.0) + 0.5);
			if (computed_refresh_rate_ms > 0)
			{
				sample_interval_gpu_ms = MAX_INT(computed_refresh_rate_ms * GPU_REFRESH_RATE_MULTIPLIER, min_sample_interval_gpu_ms);
				_STARPU_DISP("STARPU_ENERGY_GPU_INTERVAL=%d (set by energy-reader)\n", sample_interval_gpu_ms);
			}
			else
			{
				sample_interval_gpu_ms = backup_sample_interval_gpu_ms;
				_STARPU_DISP("STARPU_ENERGY_GPU_INTERVAL=%d (WARNING : backup value)\n", sample_interval_gpu_ms);
			}
		}
	}

	return 0;
}

static int _starpu_update_energy_reader_set(int workerid, energy_samples_set_s *set)
{
	int r = energy_reader_update_readings(set, context_g);
	if (r != 0)
	{
		_STARPU_DISP("Error updating energy readings for set %d\n", set->counter_mask);
		return -1;
	}
	for (int i = 0; i < set->nb_samples; i++)
	{
		energy_sample_s *sample = &set->samples[i];
		double energy = sample->energy_total;
		unsigned long energy_ul = DOUBLE_TO_UL_WITH_PRECISION(energy, sample->decimal_precision);
		double delay_us = sample->delay_us;
		unsigned long delay_ns = (unsigned long)(delay_us * 1000.0 + 0.5);
		_starpu_trace_energy_reading_measurement((unsigned long)sample->counter,
							 (unsigned long)sample->scope,
							 (unsigned long)sample->scope_id,
							 energy_ul,
							 (unsigned long)sample->decimal_precision,
							 delay_ns,
							 workerid);
	}
	return 0;
}

#endif /* STARPU_HAVE_ENERGYREADER */

void _starpu_energyreader_init(struct _starpu_machine_config starpu_config_arg)
{
#ifdef STARPU_HAVE_ENERGYREADER
	energy_reader_enabled = _starpu_energyreader_is_enabled();
	if (!energy_reader_enabled)
	{
		return;
	}
	hwloc_topology_t hwloc_topo = starpu_get_hwloc_topology();
	context_g = energy_reader_initialize_with_options(BACKEND_MASK_ALL, hwloc_topo);
	if (!context_g)
	{
		_STARPU_DISP("Warning: energy reader initialization failed\n");
		return;
	}

	// Register energy-reader topology with fxt for later matching with workers
	struct energy_reader_topology *topo = energy_reader_get_topology(context_g);

	for (int i = 0; i < topo->nb_packages; i++)
	{
		int package_id = topo->pkg_topologies[i].os_id;
		int nb_cores = topo->pkg_topologies[i].num_cores;
		_starpu_trace_energy_reading_pkg_register(package_id, nb_cores);
	}
	for (int i = 0; i < topo->nb_cores; i++)
	{
		int core_id = topo->core_topologies[i].os_id;
		int core_pkg_id = topo->core_topologies[i].pkg_os_id;
		_starpu_trace_energy_reading_core_register(core_id, core_pkg_id);
	}
	if (topo->nb_gpus > 0)
	{
		for (unsigned int i = 0; i < topo->nb_gpus; i++)
		{
			_starpu_trace_energy_reading_gpu_register(topo->gpu_topologies[i].device_id,
								  topo->gpu_topologies[i].numa_node_id);
		}
	}

	// Register worker core / package / gpu with fxt
	int nbworkers = starpu_config_arg.topology.nworkers;
	for (int worker = 0; worker < nbworkers; worker++)
	{
		struct _starpu_worker *worker_struct = _starpu_get_worker_struct(worker);
		int workerid = worker_struct->workerid;
		enum starpu_worker_archtype archtype = worker_struct->arch;
		unsigned int memory_node = worker_struct->memory_node;
		enum starpu_node_kind node_kind = starpu_worker_get_memory_node_kind(archtype);
		hwloc_obj_t hwloc_object = starpu_worker_get_hwloc_obj(workerid);
		assert(hwloc_object != NULL);
		assert(hwloc_object->type == HWLOC_OBJ_PU);
		hwloc_obj_t core = hwloc_get_ancestor_obj_by_type(starpu_config_arg.topology.hwtopology, HWLOC_OBJ_CORE, hwloc_object);
		assert(core != NULL);
		int core_logical_index = core->logical_index;
		hwloc_obj_t package = hwloc_get_ancestor_obj_by_type(starpu_config_arg.topology.hwtopology, HWLOC_OBJ_PACKAGE, hwloc_object);
		assert(package != NULL);
		int package_logical_index = package->logical_index;
		int gpu_devid = worker_struct->devid;
		switch (archtype)
		{
		case STARPU_CPU_WORKER:
			_starpu_trace_energy_reading_register_cpu_worker(workerid, core_logical_index, package_logical_index, memory_node, node_kind);
			break;
		// Assuming either CUDA or HIP, not both
		case STARPU_CUDA_WORKER:
			_starpu_trace_energy_reading_register_cuda_worker(workerid, core_logical_index, package_logical_index, memory_node, node_kind, gpu_devid);
			break;
		case STARPU_HIP_WORKER:
			_starpu_trace_energy_reading_register_hip_worker(workerid, core_logical_index, package_logical_index, memory_node, node_kind, gpu_devid);
			break;
		default:
			continue;
		}
	}

	if (_starpu_initialize_energy_samples() != 0)
	{
		_STARPU_DISP("failed to initialize energy_reader");
		return;
	}

	STARPU_PTHREAD_MUTEX_INIT(&energy_mutex, NULL);
	_starpu_energy_clock_init();

	if (energy_reader_start_readings(context_g) != 0)
	{
		_STARPU_DISP("Warning: energy reader start readings failed\n");
	}
#else
	(void)starpu_config_arg;
#endif /* STARPU_HAVE_ENERGYREADER */
	return;
}

int _starpu_energyreader_try_measurement(int workerid)
{
#ifdef STARPU_HAVE_ENERGYREADER
	if (!energy_reader_enabled)
	{
		return 0;
	}
	if (STARPU_PTHREAD_MUTEX_TRYLOCK(&energy_mutex) == 0)
	{
		struct timespec current_time;
		_starpu_clock_gettime(&current_time);
		long time_diff_rapl_ms = TIME_DIFF_MS(last_reading_time_rapl, current_time);
		long time_diff_gpu_ms = TIME_DIFF_MS(last_reading_time_gpu, current_time);
		long time_diff_cray_ms = TIME_DIFF_MS(last_reading_time_cray,
						      current_time);

		if (rapl_set && time_diff_rapl_ms >= sample_interval_rapl_ms)
		{
			_starpu_trace_start_energy_measuring();
			int r = _starpu_update_energy_reader_set(workerid, rapl_set);
			_starpu_trace_end_energy_measuring();
			if (r != 0)
			{
				_STARPU_DISP("Error querying RAPL energy readings\n");
				STARPU_PTHREAD_MUTEX_UNLOCK(&energy_mutex);
				return -1;
			}
			_starpu_clock_gettime(&last_reading_time_rapl);
		}

		// Try updating cray readings
		if (cray_set && time_diff_cray_ms >= sample_interval_cray_ms)
		{
			_starpu_trace_start_energy_measuring();
			int r = _starpu_update_energy_reader_set(workerid, cray_set);
			_starpu_trace_end_energy_measuring();
			if (r != 0)
			{
				_STARPU_DISP("Error querying CRAY energy readings\n");
				STARPU_PTHREAD_MUTEX_UNLOCK(&energy_mutex);
				return -1;
			}
			_starpu_clock_gettime(&last_reading_time_cray);
		}

		// Try updating GPU readings
		if (gpu_set && time_diff_gpu_ms >= sample_interval_gpu_ms)
		{
			_starpu_trace_start_energy_measuring();
			int r = _starpu_update_energy_reader_set(workerid, gpu_set);
			_starpu_trace_end_energy_measuring();
			if (r != 0)
			{
				_STARPU_DISP("Error querying GPU energy readings\n");
				STARPU_PTHREAD_MUTEX_UNLOCK(&energy_mutex);
				return -1;
			}
			_starpu_clock_gettime(&last_reading_time_gpu);
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&energy_mutex);
	}
#else
	(void)workerid;
#endif /* STARPU_HAVE_ENERGYREADER */
	return 0;
}

void _starpu_energyreader_terminate(void)
{
#ifdef STARPU_HAVE_ENERGYREADER
	if (!energy_reader_enabled)
	{
		return;
	}
	if (rapl_set)
	{
		energy_reader_print_sample_set(rapl_set);
		energy_reader_destroy_sample_set(rapl_set);
	}
	if (cray_set)
	{
		energy_reader_print_sample_set(cray_set);
		energy_reader_destroy_sample_set(cray_set);
	}
	if (gpu_set)
	{
		energy_reader_print_sample_set(gpu_set);
		energy_reader_destroy_sample_set(gpu_set);
	}
	energy_reader_shutdown(context_g);
#endif /* STARPU_HAVE_ENERGYREADER */
	return;
}
