/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <starpu.h>
#ifdef STARPU_PAPI
#include <papi.h>
#endif
#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif
#include <starpu_perfmodel.h>
#include <starpu_profiling.h>
#include <common/config.h>
#include <common/utils.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <sys/stat.h>
#include <core/perfmodel/perfmodel.h>
#include <core/jobs.h>
#include <core/workers.h>
#include <datawizard/datawizard.h>
#include <core/task.h>

/* Driver porters: adding your driver here is optional, only needed for the support of energy profiling.  */
#ifdef STARPU_USE_CUDA
#ifdef HAVE_NVMLDEVICEGETTOTALENERGYCONSUMPTION
#include <nvml.h>
#include <cuda.h>
#include <cuda_runtime.h>
#endif
#endif

#define ERROR_RETURN(retval, function) do { PAPI_perror(function); fprintf(stderr, "Error %d %s:line %d\n", retval,__FILE__,__LINE__);  return(retval); } while (0)

#if 0
#define debug(fmt, ...) printf(fmt, ## __VA_ARGS__)
#else
#define debug(fmt, ...)
#endif

#ifdef STARPU_PAPI
#ifdef STARPU_HAVE_HWLOC
static const int N_EVTS = 2;
static int n_recorded;

static int nsockets;

static const char* event_names[] =
{
	"rapl::RAPL_ENERGY_PKG:cpu=%d",
	"rapl::RAPL_ENERGY_DRAM:cpu=%d"
};

static int add_event(int EventSet, int socket);

/* PAPI variables*/

/*must be initialized to PAPI_NULL before calling PAPI_create_event*/
static int EventSet = PAPI_NULL;
#endif
#endif

static double t1;

#ifdef STARPU_USE_CUDA
#ifdef HAVE_NVMLDEVICEGETTOTALENERGYCONSUMPTION
static unsigned long long energy_begin, energy_end;
static nvmlDevice_t device;
#endif
#endif

int starpu_energy_start(int workerid STARPU_ATTRIBUTE_UNUSED, enum starpu_worker_archtype archi)
{
	t1 = starpu_timing_now();

	/* Driver porters: adding your driver here is optional, only needed for the support of energy measurement.  */

	switch (archi)
	{
#ifdef STARPU_PAPI
#ifdef STARPU_HAVE_HWLOC
	case STARPU_CPU_WORKER:
	{
		STARPU_ASSERT_MSG(workerid == -1, "For CPUs we cannot measure each worker separately, use where = STARPU_CPU and leave workerid as -1\n");

		int retval, number;

		struct _starpu_machine_config *config = _starpu_get_machine_config();
		hwloc_topology_t topology = config->topology.hwtopology;

		nsockets = hwloc_get_nbobjs_by_type(topology, HWLOC_OBJ_PACKAGE);

		if ((retval = PAPI_library_init(PAPI_VER_CURRENT)) != PAPI_VER_CURRENT)
			ERROR_RETURN(retval, "PAPI_library_init");

		/* Creating the eventset */
		if ((retval = PAPI_create_eventset(&EventSet)) != PAPI_OK)
			ERROR_RETURN(retval, "PAPI_create_eventset");

		int i;
		for (i = 0 ; i < nsockets ; i ++ )
		{
			/* return the index of socket */
			hwloc_obj_t obj = hwloc_get_obj_by_type(topology, HWLOC_OBJ_PACKAGE, i);
			STARPU_ASSERT(obj);
			if ( (retval = add_event(EventSet, obj->os_index)) != PAPI_OK)
			{
				if (retval == PAPI_EPERM)
					_STARPU_DISP("PAPI could not access counters due to permissions errors. Perhaps your system requires to run measurements as root?\n");
				else if (retval == PAPI_ENOEVNT)
					_STARPU_DISP("PAPI could not access counters. Perhaps your system requires to run measurements as root?\n");
				ERROR_RETURN(retval, "PAPI_add_named_event");
			}
		}

		/* get the number of events in the event set */
		number = 0;
		if ( (retval = PAPI_list_events(EventSet, NULL, &number)) != PAPI_OK)
			ERROR_RETURN(retval, "PAPI_list_events");

		debug("There are %d events in the event set\n", number);

		/* Start counting */
		if ( (retval = PAPI_start(EventSet)) != PAPI_OK)
			ERROR_RETURN(retval, "PAPI_start");

		return retval;
	}
#endif
#endif

#ifdef HAVE_NVMLDEVICEGETTOTALENERGYCONSUMPTION
	case STARPU_CUDA_WORKER:
	{
		STARPU_ASSERT_MSG(workerid != -1, "For CUDA GPUs we measure each GPU separately, please specify a worker\n");
		int devid = starpu_worker_get_devid(workerid);
		int ret = nvmlDeviceGetHandleByIndex_v2 (devid,  &device);
		if (ret != NVML_SUCCESS)
		{
			_STARPU_DISP("Could not get CUDA device %d from nvml\n", devid);
			return -1;
		}
		ret = nvmlDeviceGetTotalEnergyConsumption ( device, &energy_begin );
		if (ret != NVML_SUCCESS)
		{
			_STARPU_DISP("Could not measure energy used by CUDA device %d\n", devid);
			return -1;
		}
		return 0;
	}
	break;
#endif

	default:
		printf("Error: worker is not supported ! \n");
		return -1;
	}
}

int starpu_energy_stop(struct starpu_perfmodel *model, struct starpu_task *task, unsigned nimpl, unsigned ntasks, int workerid, enum starpu_worker_archtype archi)
{
	double energy = 0.;
	int retval = 0;
	unsigned cpuid = 0;
	double t2 = starpu_timing_now();
	double t STARPU_ATTRIBUTE_UNUSED = t2 - t1;

	switch (archi)
	{
#ifdef STARPU_PAPI
#ifdef STARPU_HAVE_HWLOC
	case STARPU_CPU_WORKER:
	{
		STARPU_ASSERT_MSG(workerid == -1, "For CPUs we cannot measure each worker separately, use where = STARPU_CPU and leave workerid as -1\n");

		/*This is where we store the values we read from the eventset */
		long long values[nsockets*n_recorded];

		/* Stop counting and store the values into the array */
		if ( (retval = PAPI_stop(EventSet, values)) != PAPI_OK)
			ERROR_RETURN(retval, "PAPI_stop");

		int k,s;

		for( s = 0 ; s < nsockets ; s ++)
		{
			for(k = 0 ; k < n_recorded; k++)
			{
				double delta = values[s * n_recorded + k]*0.23/1.0e9;
				energy += delta;

				debug("%-40s%12.6f J\t(for %f us, Average Power %.1fW)\n",
				      event_names[k],
				      delta, t, delta/(t*1.0E-6));
			}
		}

		/*removes all events from a PAPI event set */
		if ( (retval = PAPI_cleanup_eventset(EventSet)) != PAPI_OK)
			ERROR_RETURN(retval, "PAPI_cleanup_eventset");

		/*deallocates the memory associated with an empty PAPI EventSet*/
		if ( (retval = PAPI_destroy_eventset(&EventSet)) != PAPI_OK)
			ERROR_RETURN(retval, "PAPI_destroy_eventset");

		break;
	}
#endif
#endif

#ifdef HAVE_NVMLDEVICEGETTOTALENERGYCONSUMPTION
	case STARPU_CUDA_WORKER:
	{
		STARPU_ASSERT_MSG(workerid != -1, "For CUDA GPUs we measure each GPU separately, please specify a worker\n");
		int ret = nvmlDeviceGetTotalEnergyConsumption(device, &energy_end);
		if (ret != NVML_SUCCESS)
			return -1;
		energy = (energy_end - energy_begin) / 1000.;
		debug("energy consumption on device %d is %f mJ (for %f us, Average power %0.1fW)\n", 0, energy * 1000., t, energy / (t*1.0E-6));
		break;
	}
#endif

	default:
	{
		printf("Error: worker type %d is not supported! \n", archi);
		return -1;
	}
	}


	struct starpu_perfmodel_arch *arch;
	if (workerid == -1)
		/* Just take one of them */
		workerid = starpu_worker_get_by_type(archi, 0);

	arch = starpu_worker_get_perf_archtype(workerid, STARPU_NMAX_SCHED_CTXS);

	starpu_perfmodel_update_history_n(model, task, arch, cpuid, nimpl, energy / ntasks, ntasks);

	return retval;
}

#ifdef STARPU_PAPI
#ifdef STARPU_HAVE_HWLOC
static int add_event(int eventSet, int socket)
{
	int retval, i;
	for (i = 0; i < N_EVTS; i++)
	{
		char buf[255];
		snprintf(buf, sizeof(buf), event_names[i], socket);

		/* printf("Activating multiplex\n"); */
		/* retval = PAPI_set_multiplex(eventSet); */
		/* if(retval != PAPI_OK) { */
		/*      _STARPU_DISP("cannot set multiplex\n"); */
		/*      return retval; */
		/* } */
		retval = PAPI_add_named_event(eventSet, buf);
		if (retval != PAPI_OK)
		{
			if (!strcmp(event_names[i], "rapl::RAPL_ENERGY_DRAM:cpu=%d"))
			{
				/* Ok, too bad */
				_STARPU_DISP("Note: DRAM energy measurement not available\n");
				n_recorded = i+1;
				return PAPI_OK;
			}
			_STARPU_DISP("cannot add event '%s': %d\n", buf, retval);
			return retval;
		}
	}

	n_recorded = i;
	return ( PAPI_OK );
}
#endif
#endif
