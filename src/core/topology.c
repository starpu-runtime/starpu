/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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

#include <stdlib.h>
#include <stdio.h>
#include <common/config.h>
#include <core/workers.h>
#include <core/debug.h>
#include <core/topology.h>
#include <drivers/cuda/driver_cuda.h>
#include <drivers/mic/driver_mic_source.h>
#include <drivers/scc/driver_scc_source.h>
#include <drivers/mp_common/source_common.h>
#include <drivers/opencl/driver_opencl.h>
#include <profiling/profiling.h>
#include <datawizard/memory_nodes.h>
#include <common/uthash.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#ifndef HWLOC_API_VERSION
#define HWLOC_OBJ_PU HWLOC_OBJ_PROC
#endif
#if HWLOC_API_VERSION < 0x00010b00
#define HWLOC_OBJ_NUMANODE HWLOC_OBJ_NODE
#endif

#endif

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

#ifdef STARPU_SIMGRID
#include <core/simgrid.h>
#endif

static unsigned topology_is_initialized = 0;
static int nobind;

/* For checking whether two workers share the same PU, indexed by PU number */
static int cpu_worker[STARPU_MAXCPUS];

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL) || defined(STARPU_USE_SCC) || defined(STARPU_SIMGRID)

struct handle_entry
{
	UT_hash_handle hh;
	unsigned gpuid;
};

#  if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
/* Entry in the `devices_using_cuda' hash table.  */
static struct handle_entry *devices_using_cuda;
#  endif

static unsigned may_bind_automatically[STARPU_NARCH] = { 0 };

#endif // defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)

#if defined(STARPU_HAVE_HWLOC)
static hwloc_obj_t numa_get_obj(hwloc_obj_t obj)
{
#if HWLOC_API_VERSION >= 0x00020000
	while (obj->memory_first_child == NULL)
	{
		obj = obj->parent;
		if (!obj)
			return NULL;
	}
	return obj->memory_first_child;
#else
	while (obj->type != HWLOC_OBJ_NUMANODE)
	{
		obj = obj->parent;

		/* If we don't find a "node" obj before the root, this means
		 * hwloc does not know whether there are numa nodes or not, so
		 * we should not use a per-node sampling in that case. */
		if (!obj)
			return NULL;
	}
	return obj;
#endif
}
#endif

/*
 * Discover the topology of the machine
 */

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL) || defined(STARPU_USE_SCC)  || defined(STARPU_SIMGRID)
static void
_starpu_initialize_workers_deviceid (int *explicit_workers_gpuid,
				  int *current, int *workers_gpuid,
				  const char *varname, unsigned nhwgpus,
				  enum starpu_worker_archtype type)
{
	char *strval;
	unsigned i;

	*current = 0;

	/* conf->workers_gpuid indicates the successive GPU identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_CUDAID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cores. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	strval = starpu_getenv(varname);
	if (strval)
	{
		/* STARPU_WORKERS_CUDAID certainly contains less entries than
		 * STARPU_NMAXWORKERS, so we reuse its entries in a round
		 * robin fashion: "1 2" is equivalent to "1 2 1 2 1 2 .... 1
		 * 2". */
		unsigned wrap = 0;
		unsigned number_of_entries = 0;

		char *endptr;
		/* we use the content of the STARPU_WORKERS_CUDAID
		 * env. variable */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
		{
			if (!wrap)
			{
				long int val;
				val = strtol(strval, &endptr, 10);
				if (endptr != strval)
				{
					workers_gpuid[i] = (unsigned)val;
					strval = endptr;
				}
				else
				{
					/* there must be at least one entry */
					STARPU_ASSERT(i != 0);
					number_of_entries = i;

					/* there is no more values in the
					 * string */
					wrap = 1;

					workers_gpuid[i] = workers_gpuid[0];
				}
			}
			else
			{
				workers_gpuid[i] =
					workers_gpuid[i % number_of_entries];
			}
		}
	}
	else if (explicit_workers_gpuid)
	{
		/* we use the explicit value from the user */
		memcpy(workers_gpuid,
                       explicit_workers_gpuid,
                       STARPU_NMAXWORKERS*sizeof(unsigned));
	}
	else
	{
		/* by default, we take a round robin policy */
		if (nhwgpus > 0)
		     for (i = 0; i < STARPU_NMAXWORKERS; i++)
			  workers_gpuid[i] = (unsigned)(i % nhwgpus);

		/* StarPU can use sampling techniques to bind threads
		 * correctly */
		may_bind_automatically[type] = 1;
	}
}
#endif

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
static void
_starpu_initialize_workers_cuda_gpuid (struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

        _starpu_initialize_workers_deviceid (
		uconf->use_explicit_workers_cuda_gpuid == 0
		? NULL
		: (int *)uconf->workers_cuda_gpuid,
		&(config->current_cuda_gpuid),
		(int *)topology->workers_cuda_gpuid,
		"STARPU_WORKERS_CUDAID",
		topology->nhwcudagpus,
		STARPU_CUDA_WORKER);
}

static inline int
_starpu_get_next_cuda_gpuid (struct _starpu_machine_config *config)
{
	unsigned i =
		((config->current_cuda_gpuid++) % config->topology.ncudagpus);

	return (int)config->topology.workers_cuda_gpuid[i];
}
#endif

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
static void
_starpu_initialize_workers_opencl_gpuid (struct _starpu_machine_config*config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

        _starpu_initialize_workers_deviceid(
		uconf->use_explicit_workers_opencl_gpuid == 0
		? NULL
		: (int *)uconf->workers_opencl_gpuid,
		&(config->current_opencl_gpuid),
		(int *)topology->workers_opencl_gpuid,
		"STARPU_WORKERS_OPENCLID",
		topology->nhwopenclgpus,
		STARPU_OPENCL_WORKER);

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
        // Detect devices which are already used with CUDA
        {
                unsigned tmp[STARPU_NMAXWORKERS];
                unsigned nb=0;
                int i;
                for(i=0 ; i<STARPU_NMAXWORKERS ; i++)
		{
			struct handle_entry *entry;
			int devid = config->topology.workers_opencl_gpuid[i];

			HASH_FIND_INT(devices_using_cuda, &devid, entry);
			if (entry == NULL)
			{
                                tmp[nb] = topology->workers_opencl_gpuid[i];
                                nb++;
                        }
                }
                for (i=nb ; i<STARPU_NMAXWORKERS ; i++)
			tmp[i] = -1;
                memcpy (topology->workers_opencl_gpuid, tmp,
			sizeof(unsigned)*STARPU_NMAXWORKERS);
        }
#endif /* STARPU_USE_CUDA */
        {
                // Detect identical devices
		struct handle_entry *devices_already_used = NULL;
                unsigned tmp[STARPU_NMAXWORKERS];
                unsigned nb=0;
                int i;

                for(i=0 ; i<STARPU_NMAXWORKERS ; i++)
		{
			int devid = topology->workers_opencl_gpuid[i];
			struct handle_entry *entry;
			HASH_FIND_INT(devices_already_used, &devid, entry);
			if (entry == NULL)
			{
				struct handle_entry *entry2;
				_STARPU_MALLOC(entry2, sizeof(*entry2));
				entry2->gpuid = devid;
				HASH_ADD_INT(devices_already_used, gpuid,
					     entry2);
                                tmp[nb] = devid;
                                nb ++;
                        }
                }
		struct handle_entry *entry, *tempo;
		HASH_ITER(hh, devices_already_used, entry, tempo)
		{
			HASH_DEL(devices_already_used, entry);
			free(entry);
		}
                for (i=nb ; i<STARPU_NMAXWORKERS ; i++)
			tmp[i] = -1;
                memcpy (topology->workers_opencl_gpuid, tmp,
			sizeof(unsigned)*STARPU_NMAXWORKERS);
        }
}

static inline int
_starpu_get_next_opencl_gpuid (struct _starpu_machine_config *config)
{
	unsigned i =
		((config->current_opencl_gpuid++) % config->topology.nopenclgpus);

	return (int)config->topology.workers_opencl_gpuid[i];
}
#endif

#if 0
#if defined(STARPU_USE_MIC) || defined(STARPU_SIMGRID)
static void _starpu_initialize_workers_mic_deviceid(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

	_starpu_initialize_workers_deviceid(
		uconf->use_explicit_workers_mic_deviceid == 0
		? NULL
		: (int *)config->user_conf->workers_mic_deviceid,
		&(config->current_mic_deviceid),
		(int *)topology->workers_mic_deviceid,
		"STARPU_WORKERS_MICID",
		topology->nhwmiccores,
		STARPU_MIC_WORKER);
}
#endif
#endif

#ifdef STARPU_USE_SCC
static void _starpu_initialize_workers_scc_deviceid(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = &config->conf;

	_starpu_initialize_workers_deviceid(
		uconf->use_explicit_workers_scc_deviceid == 0
		? NULL
		: (int *) uconf->workers_scc_deviceid,
		&(config->current_scc_deviceid),
		(int *)topology->workers_scc_deviceid,
		"STARPU_WORKERS_SCCID",
		topology->nhwscc,
		STARPU_SCC_WORKER);
}
#endif /* STARPU_USE_SCC */

#if 0
#ifdef STARPU_USE_MIC
static inline int _starpu_get_next_mic_deviceid(struct _starpu_machine_config *config)
{
	unsigned i = ((config->current_mic_deviceid++) % config->topology.nmicdevices);

	return (int)config->topology.workers_mic_deviceid[i];
}
#endif
#endif

#ifdef STARPU_USE_SCC
static inline int _starpu_get_next_scc_deviceid(struct _starpu_machine_config *config)
{
	unsigned i = ((config->current_scc_deviceid++) % config->topology.nsccdevices);

	return (int)config->topology.workers_scc_deviceid[i];
}
#endif

#ifdef STARPU_USE_MIC
static void
_starpu_init_mic_topology (struct _starpu_machine_config *config, long mic_idx)
{
	/* Discover the topology of the mic node identifier by MIC_IDX. That
	 * means, make this StarPU instance aware of the number of cores available
	 * on this MIC device. Update the `nhwmiccores' topology field
	 * accordingly. */

	struct _starpu_machine_topology *topology = &config->topology;

	int nbcores;
	_starpu_src_common_sink_nbcores (mic_nodes[mic_idx], &nbcores);
	topology->nhwmiccores[mic_idx] = nbcores;
}

static int
_starpu_init_mic_node (struct _starpu_machine_config *config, int mic_idx,
		       COIENGINE *coi_handle, COIPROCESS *coi_process)
{
	/* Initialize the MIC node of index MIC_IDX. */

	struct starpu_conf *user_conf = &config->conf;

	char ***argv = _starpu_get_argv();
	const char *suffixes[] = {"-mic", "_mic", NULL};

	/* Environment variables to send to the Sink, it informs it what kind
	 * of node it is (architecture and type) as there is no way to discover
	 * it itself */
	char mic_idx_env[32];
	sprintf(mic_idx_env, "_STARPU_MIC_DEVID=%d", mic_idx);

	/* XXX: this is currently necessary so that the remote process does not
	 * segfault. */
	char nb_mic_env[32];
	sprintf(nb_mic_env, "_STARPU_MIC_NB=%d", 2);

	const char *mic_sink_env[] = {"STARPU_SINK=STARPU_MIC", mic_idx_env, nb_mic_env, NULL};

	char mic_sink_program_path[1024];
	/* Let's get the helper program to run on the MIC device */
	int mic_file_found =
	    _starpu_src_common_locate_file (mic_sink_program_path,
					    sizeof(mic_sink_program_path),
					    starpu_getenv("STARPU_MIC_SINK_PROGRAM_NAME"),
					    starpu_getenv("STARPU_MIC_SINK_PROGRAM_PATH"),
					    user_conf->mic_sink_program_path,
					    (argv ? (*argv)[0] : NULL),
					    suffixes);

	if (0 != mic_file_found)
	{
		_STARPU_MSG("No MIC program specified, use the environment\n"
			    "variable STARPU_MIC_SINK_PROGRAM_NAME or the environment\n"
			    "or the field 'starpu_conf.mic_sink_program_path'\n"
			    "to define it.\n");

		return -1;
	}

	COIRESULT res;
	/* Let's get the handle which let us manage the remote MIC device */
	res = COIEngineGetHandle(COI_ISA_MIC, mic_idx, coi_handle);
	if (STARPU_UNLIKELY(res != COI_SUCCESS))
		STARPU_MIC_SRC_REPORT_COI_ERROR(res);

	/* We launch the helper on the MIC device, which will wait for us
	 * to give it work to do.
	 * As we will communicate further with the device throught scif we
	 * don't need to keep the process pointer */
	res = COIProcessCreateFromFile(*coi_handle, mic_sink_program_path, 0, NULL, 0,
				       mic_sink_env, 1, NULL, 0, NULL,
				       coi_process);
	if (STARPU_UNLIKELY(res != COI_SUCCESS))
		STARPU_MIC_SRC_REPORT_COI_ERROR(res);

	/* Let's create the node structure, we'll communicate with the peer
	 * through scif thanks to it */
	mic_nodes[mic_idx] =
		_starpu_mp_common_node_create(STARPU_MIC_SOURCE, mic_idx);

	return 0;
}
#endif

#ifndef STARPU_SIMGRID
#ifdef STARPU_HAVE_HWLOC
static void
_starpu_allocate_topology_userdata(hwloc_obj_t obj)
{
	unsigned i;

	_STARPU_CALLOC(obj->userdata, 1, sizeof(struct _starpu_hwloc_userdata));
	for (i = 0; i < obj->arity; i++)
		_starpu_allocate_topology_userdata(obj->children[i]);
#if HWLOC_API_VERSION >= 0x00020000
	hwloc_obj_t child;
	for (child = obj->io_first_child; child; child = child->next_sibling)
		_starpu_allocate_topology_userdata(child);
#endif
}

static void
_starpu_deallocate_topology_userdata(hwloc_obj_t obj)
{
	unsigned i;
	struct _starpu_hwloc_userdata *data = obj->userdata;

	STARPU_ASSERT(!data->worker_list || data->worker_list == (void*)-1);
	free(data);
	for (i = 0; i < obj->arity; i++)
		_starpu_deallocate_topology_userdata(obj->children[i]);
#if HWLOC_API_VERSION >= 0x00020000
	hwloc_obj_t child;
	for (child = obj->io_first_child; child; child = child->next_sibling)
		_starpu_deallocate_topology_userdata(child);
#endif
}
#endif
#endif

static void
_starpu_init_topology (struct _starpu_machine_config *config)
{
	/* Discover the topology, meaning finding all the available PUs for
	   the compiled drivers. These drivers MUST have been initialized
	   before calling this function. The discovered topology is filled in
	   CONFIG. */

	struct _starpu_machine_topology *topology = &config->topology;

	if (topology_is_initialized)
		return;

	nobind = starpu_get_env_number("STARPU_WORKERS_NOBIND");

	topology->nhwcpus = 0;
	topology->nhwpus = 0;

#ifndef STARPU_SIMGRID
#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_init(&topology->hwtopology);
	char *hwloc_input = starpu_getenv("STARPU_HWLOC_INPUT");
	if (hwloc_input && hwloc_input[0])
		hwloc_topology_set_xml(topology->hwtopology, hwloc_input);
	_starpu_topology_filter(topology->hwtopology);
	hwloc_topology_load(topology->hwtopology);
	_starpu_allocate_topology_userdata(hwloc_get_root_obj(topology->hwtopology));
#endif
#endif

#ifdef STARPU_SIMGRID
	config->topology.nhwcpus = config->topology.nhwpus = _starpu_simgrid_get_nbhosts("CPU");
#elif defined(STARPU_HAVE_HWLOC)
	/* Discover the CPUs relying on the hwloc interface and fills CONFIG
	 * accordingly. */


	config->cpu_depth = hwloc_get_type_depth (topology->hwtopology,
						  HWLOC_OBJ_CORE);
	config->pu_depth = hwloc_get_type_depth (topology->hwtopology,
					     HWLOC_OBJ_PU);

	/* Would be very odd */
	STARPU_ASSERT(config->cpu_depth != HWLOC_TYPE_DEPTH_MULTIPLE);

	if (config->cpu_depth == HWLOC_TYPE_DEPTH_UNKNOWN)
	{
		/* unknown, using logical procesors as fallback */
		_STARPU_DISP("Warning: The OS did not report CPU cores. Assuming there is only one hardware thread per core.\n");
		config->cpu_depth = hwloc_get_type_depth(topology->hwtopology,
							 HWLOC_OBJ_PU);
	}

	topology->nhwcpus = hwloc_get_nbobjs_by_depth (topology->hwtopology,
						       config->cpu_depth);
	topology->nhwpus = hwloc_get_nbobjs_by_depth (topology->hwtopology,
						       config->pu_depth);

#elif defined(HAVE_SYSCONF)
	/* Discover the CPUs relying on the sysconf(3) function and fills
	 * CONFIG accordingly. */

	config->topology.nhwcpus = config->topology.nhwpus = sysconf(_SC_NPROCESSORS_ONLN);

#elif defined(_WIN32)
	/* Discover the CPUs on Cygwin and MinGW systems. */

	SYSTEM_INFO sysinfo;
	GetSystemInfo(&sysinfo);
	config->topology.nhwcpus = config->topology.nhwpus = sysinfo.dwNumberOfProcessors;
#else
#warning no way to know number of cores, assuming 1
	config->topology.nhwcpus = config->topology.nhwpus = 1;
#endif

	if (config->conf.ncuda != 0)
		_starpu_cuda_discover_devices(config);
	if (config->conf.nopencl != 0)
		_starpu_opencl_discover_devices(config);
#ifdef STARPU_USE_SCC
	config->topology.nhwscc = _starpu_scc_src_get_device_count();
#endif

	topology_is_initialized = 1;
}

/*
 * Bind workers on the different processors
 */
static void
_starpu_initialize_workers_bindid (struct _starpu_machine_config *config)
{
	char *strval;
	unsigned i;

	struct _starpu_machine_topology *topology = &config->topology;

	config->current_bindid = 0;

	/* conf->workers_bindid indicates the successive logical PU identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cores. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	strval = starpu_getenv("STARPU_WORKERS_CPUID");
	if (strval)
	{
		/* STARPU_WORKERS_CPUID certainly contains less entries than
		 * STARPU_NMAXWORKERS, so we reuse its entries in a round
		 * robin fashion: "1 2" is equivalent to "1 2 1 2 1 2 .... 1
		 * 2". */
		unsigned wrap = 0;
		unsigned number_of_entries = 0;

		char *endptr;
		/* we use the content of the STARPU_WORKERS_CPUID
		 * env. variable */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
		{
			if (!wrap)
			{
				long int val;
				val = strtol(strval, &endptr, 10);
				if (endptr != strval)
				{
					topology->workers_bindid[i] =
						(unsigned)(val % topology->nhwpus);
					strval = endptr;
					if (*strval == '-')
					{
						/* range of values */
						long int endval;
						strval++;
						if (*strval && *strval != ' ' && *strval != ',')
						{
							endval = strtol(strval, &endptr, 10);
							strval = endptr;
						}
						else
						{
							endval = topology->nhwpus-1;
							if (*strval)
								strval++;
						}
						for (val++; val <= endval && i < STARPU_NMAXWORKERS-1; val++)
						{
							i++;
							topology->workers_bindid[i] =
								(unsigned)(val % topology->nhwpus);
						}
					}
					if (*strval == ',')
						strval++;
				}
				else
				{
					/* there must be at least one entry */
					STARPU_ASSERT(i != 0);
					number_of_entries = i;

					/* there is no more values in the
					 * string */
					wrap = 1;

					topology->workers_bindid[i] =
						topology->workers_bindid[0];
				}
			}
			else
			{
				topology->workers_bindid[i] =
					topology->workers_bindid[i % number_of_entries];
			}
		}
	}
	else if (config->conf.use_explicit_workers_bindid)
	{
		/* we use the explicit value from the user */
		memcpy(topology->workers_bindid,
			config->conf.workers_bindid,
			STARPU_NMAXWORKERS*sizeof(unsigned));
	}
	else
	{
		int nth_per_core = starpu_get_env_number_default("STARPU_NTHREADS_PER_CORE", 1);
		int k;
		int nbindids=0;
		int nhyperthreads = topology->nhwpus / topology->nhwcpus;
		STARPU_ASSERT_MSG(nth_per_core > 0 && nth_per_core <= nhyperthreads , "Incorrect number of hyperthreads");

		i = 0; /* PU number currently assigned */
		k = 0; /* Number of threads already put on the current core */
		while(nbindids < STARPU_NMAXWORKERS)
		{
			if (k >= nth_per_core)
			{
				/* We have already put enough workers on this
				 * core, skip remaining PUs from this core, and
				 * proceed with next core */
				i += nhyperthreads-nth_per_core;
				k = 0;
				continue;
			}

			/* Add a worker to this core, by using this logical PU */
			topology->workers_bindid[nbindids++] =
				(unsigned)(i % topology->nhwpus);
			k++;
			i++;
		}
	}

	for (i = 0; i < STARPU_MAXCPUS;i++)
		cpu_worker[i] = STARPU_NOWORKERID;

	/* no binding yet */
	memset(&config->currently_bound, 0, sizeof(config->currently_bound));
}

/* This function gets the identifier of the next core on which to bind a
 * worker. In case a list of preferred cores was specified (logical indexes),
 * we look for a an available core among the list if possible, otherwise a
 * round-robin policy is used. */
static inline int
_starpu_get_next_bindid (struct _starpu_machine_config *config,
			 int *preferred_binding, int npreferred)
{
	struct _starpu_machine_topology *topology = &config->topology;

	int current_preferred;
	int nhyperthreads = topology->nhwpus / topology->nhwcpus;
	unsigned i;

	if (npreferred)
	{
		STARPU_ASSERT_MSG(preferred_binding, "Passing NULL pointer for parameter preferred_binding with a non-0 value of parameter npreferred");
	}

	/* loop over the preference list */
	for (current_preferred = 0;
	     current_preferred < npreferred;
	     current_preferred++)
	{
		/* Try to get this core */
		unsigned requested_core = preferred_binding[current_preferred];
		unsigned requested_bindid = requested_core * nhyperthreads;

		/* can we bind the worker on the preferred core ? */
		unsigned ind;
		/* Look at the remaining cores to be bound to */
		for (ind = 0;
		     ind < topology->nhwpus / nhyperthreads;
		     ind++)
		{
			if (topology->workers_bindid[ind] == requested_bindid && !config->currently_bound[ind])
			{
				/* the cpu is available, we use it ! */
				config->currently_bound[ind] = 1;
				return requested_bindid;
			}
		}
	}

	for (i = config->current_bindid; i < topology->nhwpus / nhyperthreads; i++)
		if (!config->currently_bound[i])
			/* Found a cpu ready for use, use it! */
			break;

	if (i == topology->nhwpus / nhyperthreads)
	{
		/* Finished binding on all cpus, restart from start in
		 * case the user really wants overloading */
		memset(&config->currently_bound, 0, sizeof(config->currently_bound));
		i = 0;
	}

	STARPU_ASSERT(i < topology->nhwpus / nhyperthreads);
	int bindid = topology->workers_bindid[i];
	config->currently_bound[i] = 1;
	config->current_bindid = i;
	return bindid;
}

unsigned
_starpu_topology_get_nhwcpu (struct _starpu_machine_config *config)
{
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	return config->topology.nhwcpus;
}

unsigned
_starpu_topology_get_nhwpu (struct _starpu_machine_config *config)
{
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	return config->topology.nhwpus;
}

#ifdef STARPU_HAVE_HWLOC
void _starpu_topology_filter(hwloc_topology_t topology)
{
#if HWLOC_API_VERSION >= 0x20000
	hwloc_topology_set_io_types_filter(topology, HWLOC_TYPE_FILTER_KEEP_IMPORTANT);
	hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM);
#else
	hwloc_topology_set_flags(topology, HWLOC_TOPOLOGY_FLAG_IS_THISSYSTEM | HWLOC_TOPOLOGY_FLAG_IO_DEVICES | HWLOC_TOPOLOGY_FLAG_IO_BRIDGES);
#endif
#ifdef HAVE_HWLOC_TOPOLOGY_SET_COMPONENTS
#  ifndef STARPU_USE_CUDA
	hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "cuda");
	hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "nvml");
#  endif
#  ifndef STARPU_USE_OPENCL
	hwloc_topology_set_components(topology, HWLOC_TOPOLOGY_COMPONENTS_FLAG_BLACKLIST, "opencl");
#  endif
#endif
}
#endif

#ifdef STARPU_USE_MIC
static void
_starpu_init_mic_config (struct _starpu_machine_config *config,
			 struct starpu_conf *user_conf,
			 unsigned mic_idx)
{
	// Configure the MIC device of index MIC_IDX.

	struct _starpu_machine_topology *topology = &config->topology;

	topology->nhwmiccores[mic_idx] = 0;

	_starpu_init_mic_topology (config, mic_idx);

	int nmiccores;
	nmiccores = starpu_get_env_number("STARPU_NMICTHREADS");

	STARPU_ASSERT_MSG(nmiccores >= -1, "nmiccores can not be negative and different from -1 (is is %d)", nmiccores);
	if (nmiccores == -1)
	{
		/* Nothing was specified, so let's use the number of
		 * detected mic cores. ! */
		nmiccores = topology->nhwmiccores[mic_idx];
	}
	else
	{
		if ((unsigned) nmiccores > topology->nhwmiccores[mic_idx])
		{
			/* The user requires more MIC cores than there is available */
			_STARPU_MSG("# Warning: %d MIC cores requested. Only %u available.\n", nmiccores, topology->nhwmiccores[mic_idx]);
			nmiccores = topology->nhwmiccores[mic_idx];
		}
	}

	topology->nmiccores[mic_idx] = nmiccores;
	STARPU_ASSERT_MSG(topology->nmiccores[mic_idx] + topology->nworkers <= STARPU_NMAXWORKERS,
			  "topology->nmiccores[mic_idx(%u)] (%u) + topology->nworkers (%u) <= STARPU_NMAXWORKERS (%d)",
			  mic_idx, topology->nmiccores[mic_idx], topology->nworkers, STARPU_NMAXWORKERS);

	/* _starpu_initialize_workers_mic_deviceid (config); */

	unsigned miccore_id;
	for (miccore_id = 0; miccore_id < topology->nmiccores[mic_idx]; miccore_id++)
	{
		int worker_idx = topology->nworkers + miccore_id;
		config->workers[worker_idx].arch = STARPU_MIC_WORKER;
		_STARPU_MALLOC(config->workers[worker_idx].perf_arch.devices, sizeof(struct starpu_perfmodel_device));
		config->workers[worker_idx].perf_arch.ndevices = 1;
		config->workers[worker_idx].perf_arch.devices[0].type = STARPU_MIC_WORKER;
		config->workers[worker_idx].perf_arch.devices[0].devid = mic_idx;
		config->workers[worker_idx].perf_arch.devices[0].ncores = 1;
		config->workers[worker_idx].devid = mic_idx;
		config->workers[worker_idx].subworkerid = miccore_id;
		config->workers[worker_idx].worker_mask = STARPU_MIC;
		config->worker_mask |= STARPU_MIC;
	}

	topology->nworkers += topology->nmiccores[mic_idx];
    }

#ifdef STARPU_USE_MIC
static COIENGINE mic_handles[STARPU_MAXMICDEVS];
COIPROCESS _starpu_mic_process[STARPU_MAXMICDEVS];
#endif

static void
_starpu_init_mp_config (struct _starpu_machine_config *config,
			struct starpu_conf *user_conf)
{
	/* Discover and configure the mp topology. That means:
	 * - discover the number of mp nodes;
	 * - initialize each discovered node;
	 * - discover the local topology (number of PUs/devices) of each node;
	 * - configure the workers accordingly.
	 */

	struct _starpu_machine_topology *topology = &config->topology;

	// We currently only support MIC at this level.
#ifdef STARPU_USE_MIC

	/* Discover and initialize the number of MIC nodes through the mp
	 * infrastructure. */
	unsigned nhwmicdevices = _starpu_mic_src_get_device_count();

	int reqmicdevices = starpu_get_env_number("STARPU_NMIC");
	if (reqmicdevices == -1 && user_conf)
		reqmicdevices = user_conf->nmic;
	if (reqmicdevices == -1)
		reqmicdevices = nhwmicdevices;

	STARPU_ASSERT_MSG(reqmicdevices >= -1, "nmic can not be negative and different from -1 (is is %d)", reqmicdevices);
	if (reqmicdevices == -1)
	{
		/* Nothing was specified, so let's use the number of
		 * detected mic devices. ! */
		reqmicdevices = nhwmicdevices;
	}
	else
	{
		if ((unsigned) reqmicdevices > nhwmicdevices)
		{
			/* The user requires more MIC devices than there is available */
			_STARPU_MSG("# Warning: %d MIC devices requested. Only %u available.\n", reqmicdevices, nhwmicdevices);
			reqmicdevices = nhwmicdevices;
		}
	}

	topology->nmicdevices = 0;
	unsigned i;
	for (i = 0; i < (unsigned) reqmicdevices; i++)
		if (0 == _starpu_init_mic_node (config, i, &mic_handles[i], &_starpu_mic_process[i]))
			topology->nmicdevices++;


	for (i = 0; i < topology->nmicdevices; i++)
		_starpu_init_mic_config (config, user_conf, i);
#endif
}

static void
_starpu_deinit_mic_node (unsigned mic_idx)
{
	_starpu_mp_common_send_command(mic_nodes[mic_idx], STARPU_EXIT, NULL, 0);

	COIProcessDestroy(_starpu_mic_process[mic_idx], -1, 0, NULL, NULL);

	_starpu_mp_common_node_destroy(mic_nodes[mic_idx]);
}

static void
_starpu_deinit_mp_config (struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	unsigned i;

	for (i = 0; i < topology->nmicdevices; i++)
		_starpu_deinit_mic_node (i);
	_starpu_mic_clear_kernels();
}
#endif

static int
_starpu_init_machine_config(struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED)
{
	int i;
	for (i = 0; i < STARPU_NMAXWORKERS; i++)
		config->workers[i].workerid = i;

	struct _starpu_machine_topology *topology = &config->topology;

	topology->nworkers = 0;
	topology->ncombinedworkers = 0;
	topology->nsched_ctxs = 0;

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	_starpu_initialize_workers_bindid(config);

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	int ncuda = config->conf.ncuda;
	int nworker_per_cuda = starpu_get_env_number_default("STARPU_NWORKER_PER_CUDA", 1);

	STARPU_ASSERT_MSG(nworker_per_cuda > 0, "STARPU_NWORKER_PER_CUDA has to be > 0");
	STARPU_ASSERT_MSG(nworker_per_cuda < STARPU_NMAXWORKERS, "STARPU_NWORKER_PER_CUDA (%d) cannot be higher than STARPU_NMAXWORKERS (%d)\n", nworker_per_cuda, STARPU_NMAXWORKERS);

#ifndef STARPU_NON_BLOCKING_DRIVERS
	if (nworker_per_cuda > 1)
	{
		_STARPU_DISP("Warning: reducing STARPU_NWORKER_PER_CUDA to 1 because blocking drivers are enabled\n");
		nworker_per_cuda = 1;
	}
#endif

	if (ncuda != 0)
	{
		/* The user did not disable CUDA. We need to initialize CUDA
 		 * early to count the number of devices */
		_starpu_init_cuda();
		int nb_devices = _starpu_get_cuda_device_count();

		STARPU_ASSERT_MSG(ncuda >= -1, "ncuda can not be negative and different from -1 (is is %d)", ncuda);
		if (ncuda == -1)
		{
			/* Nothing was specified, so let's choose ! */
			ncuda = nb_devices;
		}
		else
		{
			if (ncuda > nb_devices)
			{
				/* The user requires more CUDA devices than
				 * there is available */
				_STARPU_DISP("Warning: %d CUDA devices requested. Only %d available.\n", ncuda, nb_devices);
				ncuda = nb_devices;
			}
		}
	}

	/* Now we know how many CUDA devices will be used */
	topology->ncudagpus = ncuda;
	topology->nworkerpercuda = nworker_per_cuda;
	STARPU_ASSERT(topology->ncudagpus <= STARPU_MAXCUDADEVS);

	_starpu_initialize_workers_cuda_gpuid(config);

	unsigned cudagpu;
	for (cudagpu = 0; cudagpu < topology->ncudagpus; cudagpu++)
	{
		int devid = _starpu_get_next_cuda_gpuid(config);
		for (i = 0; i < nworker_per_cuda; i++)
		{
			int worker_idx = topology->nworkers + cudagpu * nworker_per_cuda + i;

			config->workers[worker_idx].arch = STARPU_CUDA_WORKER;
			_STARPU_MALLOC(config->workers[worker_idx].perf_arch.devices, sizeof(struct starpu_perfmodel_device));
			config->workers[worker_idx].perf_arch.ndevices = 1;
			config->workers[worker_idx].perf_arch.devices[0].type = STARPU_CUDA_WORKER;
			config->workers[worker_idx].perf_arch.devices[0].devid = devid;
			// TODO: fix perfmodels etc.
			//config->workers[worker_idx].perf_arch.ncore = nworker_per_cuda - 1;
			config->workers[worker_idx].perf_arch.devices[0].ncores = 1;
			config->workers[worker_idx].devid = devid;
			config->workers[worker_idx].subworkerid = i;
			config->workers[worker_idx].worker_mask = STARPU_CUDA;
			config->worker_mask |= STARPU_CUDA;

			struct handle_entry *entry;
			HASH_FIND_INT(devices_using_cuda, &devid, entry);
			if (!entry)
			{
				_STARPU_MALLOC(entry, sizeof(*entry));
				entry->gpuid = devid;
				HASH_ADD_INT(devices_using_cuda, gpuid, entry);
			}
		}
        }

	topology->nworkers += topology->ncudagpus * nworker_per_cuda;
#endif

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	int nopencl = config->conf.nopencl;

	if (nopencl != 0)
	{
		/* The user did not disable OPENCL. We need to initialize
 		 * OpenCL early to count the number of devices */
		_starpu_opencl_init();
		int nb_devices;
		nb_devices = _starpu_opencl_get_device_count();

		STARPU_ASSERT_MSG(nopencl >= -1, "nopencl can not be negative and different from -1 (is is %d)", nopencl);
		if (nopencl == -1)
		{
			/* Nothing was specified, so let's choose ! */
			nopencl = nb_devices;
			if (nopencl > STARPU_MAXOPENCLDEVS)
			{
				_STARPU_DISP("Warning: %d OpenCL devices available. Only %d enabled. Use configure option --enable-maxopencldadev=xxx to update the maximum value of supported OpenCL devices.\n", nb_devices, STARPU_MAXOPENCLDEVS);
				nopencl = STARPU_MAXOPENCLDEVS;
			}
		}
		else
		{
			/* Let's make sure this value is OK. */
			if (nopencl > nb_devices)
			{
				/* The user requires more OpenCL devices than
				 * there is available */
				_STARPU_DISP("Warning: %d OpenCL devices requested. Only %d available.\n", nopencl, nb_devices);
				nopencl = nb_devices;
			}
			/* Let's make sure this value is OK. */
			if (nopencl > STARPU_MAXOPENCLDEVS)
			{
				_STARPU_DISP("Warning: %d OpenCL devices requested. Only %d enabled. Use configure option --enable-maxopencldev=xxx to update the maximum value of supported OpenCL devices.\n", nopencl, STARPU_MAXOPENCLDEVS);
				nopencl = STARPU_MAXOPENCLDEVS;
			}
		}
	}

	topology->nopenclgpus = nopencl;
	STARPU_ASSERT(topology->nopenclgpus + topology->nworkers <= STARPU_NMAXWORKERS);

	_starpu_initialize_workers_opencl_gpuid(config);

	unsigned openclgpu;
	for (openclgpu = 0; openclgpu < topology->nopenclgpus; openclgpu++)
	{
		int worker_idx = topology->nworkers + openclgpu;
		int devid = _starpu_get_next_opencl_gpuid(config);
		if (devid == -1)
		{
			// There is no more devices left
			topology->nopenclgpus = openclgpu;
			break;
		}
		config->workers[worker_idx].arch = STARPU_OPENCL_WORKER;
		_STARPU_MALLOC(config->workers[worker_idx].perf_arch.devices, sizeof(struct starpu_perfmodel_device));
		config->workers[worker_idx].perf_arch.ndevices = 1;
		config->workers[worker_idx].perf_arch.devices[0].type = STARPU_OPENCL_WORKER;
		config->workers[worker_idx].perf_arch.devices[0].devid = devid;
		config->workers[worker_idx].perf_arch.devices[0].ncores = 1;
		config->workers[worker_idx].subworkerid = 0;
		config->workers[worker_idx].devid = devid;
		config->workers[worker_idx].worker_mask = STARPU_OPENCL;
		config->worker_mask |= STARPU_OPENCL;
	}

	topology->nworkers += topology->nopenclgpus;
#endif

#ifdef STARPU_USE_SCC
	int nscc = config->conf.nscc;

	unsigned nb_scc_nodes = _starpu_scc_src_get_device_count();

	if (nscc != 0)
	{
		/* The user did not disable SCC. We need to count
		 * the number of devices */
		int nb_devices = nb_scc_nodes;

		STARPU_ASSERT_MSG(nscc >= -1, "nscc can not be negative and different from -1 (is is %d)", nscc);
		if (nscc == -1)
		{
			/* Nothing was specified, so let's choose ! */
			nscc = nb_devices;
			if (nscc > STARPU_MAXSCCDEVS)
			{
				_STARPU_DISP("Warning: %d SCC devices available. Only %d enabled. Use configuration option --enable-maxsccdev=xxx to update the maximum value of supported SCC devices.\n", nb_devices, STARPU_MAXSCCDEVS);
				nscc = STARPU_MAXSCCDEVS;
			}
		}
		else
		{
			/* Let's make sure this value is OK. */
			if (nscc > nb_devices)
			{
				/* The user requires more SCC devices than there is available */
				_STARPU_DISP("Warning: %d SCC devices requested. Only %d available.\n", nscc, nb_devices);
				nscc = nb_devices;
			}
			/* Let's make sure this value is OK. */
			if (nscc > STARPU_MAXSCCDEVS)
			{
				_STARPU_DISP("Warning: %d SCC devices requested. Only %d enabled. Use configure option --enable-maxsccdev=xxx to update the maximum value of supported SCC devices.\n", nscc, STARPU_MAXSCCDEVS);
				nscc = STARPU_MAXSCCDEVS;
			}
		}
	}

	/* Now we know how many SCC devices will be used */
	topology->nsccdevices = nscc;
	STARPU_ASSERT(topology->nsccdevices + topology->nworkers <= STARPU_NMAXWORKERS);

	_starpu_initialize_workers_scc_deviceid(config);

	unsigned sccdev;
	for (sccdev = 0; sccdev < topology->nsccdevices; sccdev++)
	{
		config->workers[topology->nworkers + sccdev].arch = STARPU_SCC_WORKER;
		int devid = _starpu_get_next_scc_deviceid(config);
		_STARPU_MALLOC(config->workers[topology->nworkers + sccdev].perf_arch.devices, sizeof(struct starpu_perfmodel_device));
		config->workers[topology->nworkers + sccdev].perf_arch.ndevices = 1;

		config->workers[topology->nworkers + sccdev].perf_arch.devices[0].type = STARPU_SCC_WORKER;
		config->workers[topology->nworkers + sccdev].perf_arch.devices[0].devid = sccdev;
		config->workers[topology->nworkers + sccdev].perf_arch.devices[0].ncore = 1;
		config->workers[topology->nworkers + sccdev].subworkerid = 0;
		config->workers[topology->nworkers + sccdev].devid = devid;
		config->workers[topology->nworkers + sccdev].worker_mask = STARPU_SCC;
		config->worker_mask |= STARPU_SCC;
	}

	for (; sccdev < nb_scc_nodes; ++sccdev)
		_starpu_scc_exit_useless_node(sccdev);

	topology->nworkers += topology->nsccdevices;
#endif /* STARPU_USE_SCC */


	/* Unless not requested, we need to complete configuration with the
	 * ones of the mp nodes. */
#ifdef STARPU_USE_MIC
	if (! no_mp_config)
	    _starpu_init_mp_config (config, &config->conf);
#endif

/* we put the CPU section after the accelerator : in case there was an
 * accelerator found, we devote one cpu */
#if defined(STARPU_USE_CPU) || defined(STARPU_SIMGRID)
	int ncpu = config->conf.ncpus;

	if (ncpu != 0)
	{
		STARPU_ASSERT_MSG(ncpu >= -1, "ncpus can not be negative and different from -1 (is is %d)", ncpu);
		if (ncpu == -1)
		{
			unsigned mic_busy_cpus = 0;
			unsigned j = 0;
			for (j = 0; j < STARPU_MAXMICDEVS; j++)
				mic_busy_cpus += (topology->nmiccores[j] ? 1 : 0);

			unsigned already_busy_cpus = mic_busy_cpus + topology->ncudagpus
				+ topology->nopenclgpus + topology->nsccdevices;

			long avail_cpus = (long) topology->nhwcpus - (long) already_busy_cpus;
			if (avail_cpus < 0)
				avail_cpus = 0;
			int nth_per_core = starpu_get_env_number_default("STARPU_NTHREADS_PER_CORE", 1);
			avail_cpus *= nth_per_core;
			ncpu = STARPU_MIN(avail_cpus, STARPU_MAXCPUS);
		}
		else
		{
			if (ncpu > STARPU_MAXCPUS)
			{
				_STARPU_DISP("Warning: %d CPU devices requested. Only %d enabled. Use configure option --enable-maxcpus=xxx to update the maximum value of supported CPU devices.\n", ncpu, STARPU_MAXCPUS);
				ncpu = STARPU_MAXCPUS;
			}
		}
	}


	topology->ncpus = ncpu;
	STARPU_ASSERT(topology->ncpus + topology->nworkers <= STARPU_NMAXWORKERS);

	unsigned cpu;
	unsigned homogeneous = starpu_get_env_number_default("STARPU_PERF_MODEL_HOMOGENEOUS_CPU", 1);
	for (cpu = 0; cpu < topology->ncpus; cpu++)
	{
		int worker_idx = topology->nworkers + cpu;
		config->workers[worker_idx].arch = STARPU_CPU_WORKER;
		_STARPU_MALLOC(config->workers[worker_idx].perf_arch.devices, sizeof(struct starpu_perfmodel_device));
		config->workers[worker_idx].perf_arch.ndevices = 1;
		config->workers[worker_idx].perf_arch.devices[0].type = STARPU_CPU_WORKER;
		config->workers[worker_idx].perf_arch.devices[0].devid = homogeneous ? 0 : cpu;
		config->workers[worker_idx].perf_arch.devices[0].ncores = 1;
		config->workers[worker_idx].subworkerid = 0;
		config->workers[worker_idx].devid = cpu;
		config->workers[worker_idx].worker_mask = STARPU_CPU;
		config->worker_mask |= STARPU_CPU;
	}

	topology->nworkers += topology->ncpus;
#endif

	if (topology->nworkers == 0)
	{
                _STARPU_DEBUG("No worker found, aborting ...\n");
		return -ENODEV;
	}
	return 0;
}

void _starpu_destroy_machine_config(struct _starpu_machine_config *config)
{
	_starpu_close_debug_logfile();

	unsigned worker;
	for (worker = 0; worker < config->topology.nworkers; worker++)
	{
		struct _starpu_worker *workerarg = &config->workers[worker];
		int bindid = workerarg->bindid;
		free(workerarg->perf_arch.devices);
#ifdef STARPU_HAVE_HWLOC
		hwloc_bitmap_free(workerarg->hwloc_cpu_set);
		if (bindid != -1)
		{
			hwloc_obj_t worker_obj = hwloc_get_obj_by_depth(config->topology.hwtopology,
									config->pu_depth,
									bindid);
			struct _starpu_hwloc_userdata *data = worker_obj->userdata;
			if (data->worker_list)
			{
				_starpu_worker_list_delete(data->worker_list);
				data->worker_list = NULL;
			}
		}
#endif
		if (bindid != -1)
		{
			free(config->bindid_workers[bindid].workerids);
			config->bindid_workers[bindid].workerids = NULL;
		}
	}
	free(config->bindid_workers);
	config->bindid_workers = NULL;
	config->nbindid = 0;
	unsigned combined_worker_id;
	for(combined_worker_id=0 ; combined_worker_id < config->topology.ncombinedworkers ; combined_worker_id++)
	{
		struct _starpu_combined_worker *combined_worker = &config->combined_workers[combined_worker_id];
#ifdef STARPU_HAVE_HWLOC
		hwloc_bitmap_free(combined_worker->hwloc_cpu_set);
#endif
		free(combined_worker->perf_arch.devices);
	}

#ifdef STARPU_HAVE_HWLOC
	_starpu_deallocate_topology_userdata(hwloc_get_root_obj(config->topology.hwtopology));
	hwloc_topology_destroy(config->topology.hwtopology);
#endif

	topology_is_initialized = 0;
#ifdef STARPU_USE_CUDA
	struct handle_entry *entry, *tmp;
	HASH_ITER(hh, devices_using_cuda, entry, tmp)
	{
		HASH_DEL(devices_using_cuda, entry);
		free(entry);
	}
	devices_using_cuda = NULL;
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)
	int i;
	for (i=0; i<STARPU_NARCH; i++)
		may_bind_automatically[i] = 0;
#endif
}

void
_starpu_bind_thread_on_cpu (
	struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED,
	int cpuid STARPU_ATTRIBUTE_UNUSED, int workerid STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_SIMGRID
	return;
#else
	if (nobind > 0)
		return;
	if (cpuid < 0)
		return;

#ifdef STARPU_HAVE_HWLOC
	const struct hwloc_topology_support *support;

#ifdef STARPU_USE_OPENCL
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#ifdef STARPU_USE_CUDA
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	if (workerid != STARPU_NOWORKERID && cpuid < STARPU_MAXCPUS)
	{
		int previous = cpu_worker[cpuid];
		if (previous != STARPU_NOWORKERID && previous != workerid)
			_STARPU_DISP("Warning: both workers %d and %d are bound to the same PU %d, this will strongly degrade performance. Maybe check starpu_machine_display's output to determine what wrong binding happened. Hwloc reported %d cores and %d threads, perhaps there is misdetection between hwloc, the kernel and the BIOS, or an administrative allocation issue from e.g. the job scheduler?\n", previous, workerid, cpuid, config->topology.nhwcpus, config->topology.nhwpus);
		else
			cpu_worker[cpuid] = workerid;
	}

	support = hwloc_topology_get_support (config->topology.hwtopology);
	if (support->cpubind->set_thisthread_cpubind)
	{
		hwloc_obj_t obj =
			hwloc_get_obj_by_depth (config->topology.hwtopology,
						config->pu_depth, cpuid);
		hwloc_bitmap_t set = obj->cpuset;
		int ret;

		hwloc_bitmap_singlify(set);
		ret = hwloc_set_cpubind (config->topology.hwtopology, set,
					 HWLOC_CPUBIND_THREAD);
		if (ret)
		{
			perror("hwloc_set_cpubind");
			STARPU_ABORT();
		}
	}

#elif defined(HAVE_PTHREAD_SETAFFINITY_NP) && defined(__linux__)
	int ret;
	/* fix the thread on the correct cpu */
	cpu_set_t aff_mask;
	CPU_ZERO(&aff_mask);
	CPU_SET(cpuid, &aff_mask);

	starpu_pthread_t self = pthread_self();

	ret = pthread_setaffinity_np(self, sizeof(aff_mask), &aff_mask);
	if (ret)
	{
		const char *msg = strerror(ret);
		_STARPU_MSG("pthread_setaffinity_np: %s\n", msg);
		STARPU_ABORT();
	}

#elif defined(_WIN32)
	DWORD mask = 1 << cpuid;
	if (!SetThreadAffinityMask(GetCurrentThread(), mask))
	{
		_STARPU_ERROR("SetThreadMaskAffinity(%lx) failed\n", mask);
	}
#else
#warning no CPU binding support
#endif
#endif
}

void
_starpu_bind_thread_on_cpus (
	struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED,
	struct _starpu_combined_worker *combined_worker STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_SIMGRID
	return;
#endif
#ifdef STARPU_HAVE_HWLOC
	const struct hwloc_topology_support *support;

#ifdef STARPU_USE_OPENC
	if (config->conf.nopencl != 0)
		_starpu_opencl_init();
#endif
#ifdef STARPU_USE_CUDA
	if (config->conf.ncuda != 0)
		_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	support = hwloc_topology_get_support(config->topology.hwtopology);
	if (support->cpubind->set_thisthread_cpubind)
	{
		hwloc_bitmap_t set = combined_worker->hwloc_cpu_set;
		int ret;

		ret = hwloc_set_cpubind (config->topology.hwtopology, set,
					 HWLOC_CPUBIND_THREAD);
		if (ret)
		{
			perror("binding thread");
			STARPU_ABORT();
		}
	}
#else
#ifdef __GLIBC__
	sched_setaffinity(0,sizeof(combined_worker->cpu_set),&combined_worker->cpu_set);
#else
#  warning no parallel worker CPU binding support
#endif
#endif
}

static void
_starpu_init_workers_binding (struct _starpu_machine_config *config, int no_mp_config STARPU_ATTRIBUTE_UNUSED)
{
	/* launch one thread per CPU */
	unsigned ram_memory_node;

	/* note that even if the CPU cpu are not used, we always have a RAM
	 * node */
	/* TODO : support NUMA  ;) */
	ram_memory_node = _starpu_memory_node_register(STARPU_CPU_RAM, 0);
	STARPU_ASSERT(ram_memory_node == STARPU_MAIN_RAM);

#ifdef STARPU_SIMGRID
	char name[16];
	msg_host_t host = _starpu_simgrid_get_host_by_name("RAM");
	STARPU_ASSERT(host);
	_starpu_simgrid_memory_node_set_host(STARPU_MAIN_RAM, host);
#endif

	/* We will store all the busid of the different (src, dst)
	 * combinations in a matrix which we initialize here. */
	_starpu_initialize_busid_matrix();

	/* Each device is initialized,
	 * giving it a memory node and a core bind id.
	 */
	/* TODO: STARPU_MAXNUMANODES */
	unsigned numa_init[1] = { 1 };
	unsigned numa_memory_nodes[1] = { ram_memory_node };
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	unsigned cuda_init[STARPU_MAXCUDADEVS] = { };
	unsigned cuda_memory_nodes[STARPU_MAXCUDADEVS];
	unsigned cuda_bindid[STARPU_MAXCUDADEVS];
#endif
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	unsigned opencl_init[STARPU_MAXOPENCLDEVS] = { };
	unsigned opencl_memory_nodes[STARPU_MAXOPENCLDEVS];
	unsigned opencl_bindid[STARPU_MAXOPENCLDEVS];
#endif
#ifdef STARPU_USE_MIC
	unsigned mic_init[STARPU_MAXMICDEVS] = { };
	unsigned mic_memory_nodes[STARPU_MAXMICDEVS];
	unsigned mic_bindid[STARPU_MAXMICDEVS];
#endif
	unsigned bindid;

	for (bindid = 0; bindid < config->nbindid; bindid++)
	{
		free(config->bindid_workers[bindid].workerids);
		config->bindid_workers[bindid].workerids = NULL;
		config->bindid_workers[bindid].nworkers = 0;
	}

	unsigned worker;
	for (worker = 0; worker < config->topology.nworkers; worker++)
	{
		unsigned memory_node = -1;
		struct _starpu_worker *workerarg = &config->workers[worker];
		unsigned devid STARPU_ATTRIBUTE_UNUSED = workerarg->devid;

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL) || defined(STARPU_USE_MIC) || defined(STARPU_SIMGRID)
		/* Perhaps the worker has some "favourite" bindings  */
		int *preferred_binding = NULL;
		int npreferred = 0;
#endif

		/* select the memory node that contains worker's memory */
		switch (workerarg->arch)
		{
			case STARPU_CPU_WORKER:
			{
				/* TODO: NUMA */
				int numaid = 0;
				/* "dedicate" a cpu core to that worker */
				if (numa_init[numaid])
				{
					memory_node = numa_memory_nodes[numaid];
				}
				else
				{
					numa_init[numaid] = 1;
					memory_node = numa_memory_nodes[numaid] = _starpu_memory_node_register(STARPU_CPU_RAM, numaid);
#ifdef STARPU_SIMGRID
					snprintf(name, sizeof(name), "RAM%d", numaid);
					host = _starpu_simgrid_get_host_by_name(name);
					STARPU_ASSERT(host);
					_starpu_simgrid_memory_node_set_host(memory_node, host);
#endif
				}
				workerarg->bindid = _starpu_get_next_bindid(config, NULL, 0);
				_starpu_memory_node_add_nworkers(memory_node);
				break;
			}
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
			case STARPU_CUDA_WORKER:
#ifndef STARPU_SIMGRID
				if (may_bind_automatically[STARPU_CUDA_WORKER])
				{
					/* StarPU is allowed to bind threads automatically */
					preferred_binding = _starpu_get_cuda_affinity_vector(devid);
					npreferred = config->topology.nhwpus;
				}
#endif /* SIMGRID */
				if (cuda_init[devid])
				{
					memory_node = cuda_memory_nodes[devid];
#ifndef STARPU_SIMGRID
					workerarg->bindid = cuda_bindid[devid];
#endif /* SIMGRID */
				}
				else
				{
					cuda_init[devid] = 1;
					workerarg->bindid = cuda_bindid[devid] = _starpu_get_next_bindid(config, preferred_binding, npreferred);
					memory_node = cuda_memory_nodes[devid] = _starpu_memory_node_register(STARPU_CUDA_RAM, devid);

					_starpu_register_bus(STARPU_MAIN_RAM, memory_node);
					_starpu_register_bus(memory_node, STARPU_MAIN_RAM);
#ifdef STARPU_SIMGRID
					const char* cuda_memcpy_peer;
					snprintf(name, sizeof(name), "CUDA%u", devid);
					host = _starpu_simgrid_get_host_by_name(name);
					STARPU_ASSERT(host);
					_starpu_simgrid_memory_node_set_host(memory_node, host);
					cuda_memcpy_peer = MSG_host_get_property_value(host, "memcpy_peer");
#endif /* SIMGRID */
					if (
#ifdef STARPU_SIMGRID
						cuda_memcpy_peer && atoll(cuda_memcpy_peer)
#elif defined(STARPU_HAVE_CUDA_MEMCPY_PEER)
						1
#else /* MEMCPY_PEER */
						0
#endif /* MEMCPY_PEER */
					   )
					{
						unsigned worker2;
						for (worker2 = 0; worker2 < worker; worker2++)
						{
							struct _starpu_worker *workerarg2 = &config->workers[worker2];
							if (workerarg2->arch == STARPU_CUDA_WORKER)
							{
								unsigned memory_node2 = starpu_worker_get_memory_node(worker2);
								_starpu_register_bus(memory_node2, memory_node);
								_starpu_register_bus(memory_node, memory_node2);
							}
						}
					}
				}
				_starpu_memory_node_add_nworkers(memory_node);
				break;
#endif

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
		        case STARPU_OPENCL_WORKER:
#ifndef STARPU_SIMGRID
				if (may_bind_automatically[STARPU_OPENCL_WORKER])
				{
					/* StarPU is allowed to bind threads automatically */
					preferred_binding = _starpu_get_opencl_affinity_vector(devid);
					npreferred = config->topology.nhwpus;
				}
#endif /* SIMGRID */
				if (opencl_init[devid])
				{
					memory_node = opencl_memory_nodes[devid];
#ifndef STARPU_SIMGRID
					workerarg->bindid = opencl_bindid[devid];
#endif /* SIMGRID */
				}
				else
				{
					opencl_init[devid] = 1;
					workerarg->bindid = opencl_bindid[devid] = _starpu_get_next_bindid(config, preferred_binding, npreferred);
					memory_node = opencl_memory_nodes[devid] = _starpu_memory_node_register(STARPU_OPENCL_RAM, devid);
					_starpu_register_bus(STARPU_MAIN_RAM, memory_node);
					_starpu_register_bus(memory_node, STARPU_MAIN_RAM);
#ifdef STARPU_SIMGRID
					snprintf(name, sizeof(name), "OpenCL%u", devid);
					host = _starpu_simgrid_get_host_by_name(name);
					STARPU_ASSERT(host);
					_starpu_simgrid_memory_node_set_host(memory_node, host);
#endif /* SIMGRID */
				}
				_starpu_memory_node_add_nworkers(memory_node);
				break;
#endif

#ifdef STARPU_USE_MIC
		        case STARPU_MIC_WORKER:
				if (mic_init[devid])
				{
					memory_node = mic_memory_nodes[devid];
				}
				else
				{
					mic_init[devid] = 1;
					/* TODO */
					//if (may_bind_automatically)
					//{
					//	/* StarPU is allowed to bind threads automatically */
						//	preferred_binding = _starpu_get_mic_affinity_vector(devid);
					//	npreferred = config->topology.nhwpus;
					//}
					mic_bindid[devid] = _starpu_get_next_bindid(config, preferred_binding, npreferred);
					memory_node = mic_memory_nodes[devid] = _starpu_memory_node_register(STARPU_MIC_RAM, devid);
					_starpu_register_bus(STARPU_MAIN_RAM, memory_node);
					_starpu_register_bus(memory_node, STARPU_MAIN_RAM);

				}
				workerarg->bindid = mic_bindid[devid];
				_starpu_memory_node_add_nworkers(memory_node);
				break;
#endif /* STARPU_USE_MIC */

#ifdef STARPU_USE_SCC
			case STARPU_SCC_WORKER:
			{
				/* Node 0 represents the SCC shared memory when we're on SCC. */
				struct _starpu_memory_node_descr *descr = _starpu_memory_node_get_description();
				descr->nodes[ram_memory_node] = STARPU_SCC_SHM;

				memory_node = ram_memory_node;
				_starpu_memory_node_add_nworkers(memory_node);
			}
				break;
#endif

			default:
				STARPU_ABORT();
		}

		workerarg->memory_node = memory_node;

		_STARPU_DEBUG("worker %u type %d devid %u bound to cpu %d, STARPU memory node %u\n", worker, workerarg->arch, devid, workerarg->bindid, memory_node);

#ifdef __GLIBC__
		if (workerarg->bindid != -1)
		{
			/* Save the initial cpuset */
			CPU_ZERO(&workerarg->cpu_set);
			CPU_SET(workerarg->bindid, &workerarg->cpu_set);
		}
#endif /* __GLIBC__ */

#ifdef STARPU_HAVE_HWLOC
		if (workerarg->bindid == -1)
		{
			workerarg->hwloc_cpu_set = hwloc_bitmap_alloc();
		}
		else
		{
			/* Put the worker descriptor in the userdata field of the
			 * hwloc object describing the CPU */
			hwloc_obj_t worker_obj = hwloc_get_obj_by_depth(config->topology.hwtopology,
									config->pu_depth,
									workerarg->bindid);
			struct _starpu_hwloc_userdata *data = worker_obj->userdata;
			if (data->worker_list == NULL)
				data->worker_list = _starpu_worker_list_new();
			_starpu_worker_list_push_front(data->worker_list, workerarg);

			/* Clear the cpu set and set the cpu */
			workerarg->hwloc_cpu_set = hwloc_bitmap_dup (worker_obj->cpuset);
		}
#endif
		if (workerarg->bindid != -1)
		{
			bindid = workerarg->bindid;
			unsigned old_nbindid = config->nbindid;
			if (bindid >= old_nbindid)
			{
				/* More room needed */
				if (!old_nbindid)
					config->nbindid = STARPU_NMAXWORKERS;
				else
					config->nbindid = 2 * old_nbindid;
				if (bindid >= config->nbindid)
				{
					config->nbindid = bindid+1;
				}
				_STARPU_REALLOC(config->bindid_workers, config->nbindid * sizeof(config->bindid_workers[0]));
				memset(&config->bindid_workers[old_nbindid], 0, (config->nbindid - old_nbindid) * sizeof(config->bindid_workers[0]));
			}
			/* Add slot for this worker */
			/* Don't care about amortizing the cost, there are usually very few workers sharing the same bindid */
			config->bindid_workers[bindid].nworkers++;
			_STARPU_REALLOC(config->bindid_workers[bindid].workerids, config->bindid_workers[bindid].nworkers * sizeof(config->bindid_workers[bindid].workerids[0]));
			config->bindid_workers[bindid].workerids[config->bindid_workers[bindid].nworkers-1] = worker;
		}
	}
}

int
_starpu_build_topology (struct _starpu_machine_config *config, int no_mp_config)
{
	int ret;
	unsigned i;

	ret = _starpu_init_machine_config(config, no_mp_config);
	if (ret)
		return ret;

	/* for the data management library */
	_starpu_memory_nodes_init();

	_starpu_init_workers_binding(config, no_mp_config);

	config->cpus_nodeid = -1;
	config->cuda_nodeid = -1;
	config->opencl_nodeid = -1;
	config->mic_nodeid = -1;
	config->scc_nodeid = -1;
	for (i = 0; i < starpu_worker_get_count(); i++)
	{
		switch (starpu_worker_get_type(i))
		{
			case STARPU_CPU_WORKER:
				if (config->cpus_nodeid == -1)
					config->cpus_nodeid = starpu_worker_get_memory_node(i);
				else if (config->cpus_nodeid != (int) starpu_worker_get_memory_node(i))
					config->cpus_nodeid = -2;
				break;
			case STARPU_CUDA_WORKER:
				if (config->cuda_nodeid == -1)
					config->cuda_nodeid = starpu_worker_get_memory_node(i);
				else if (config->cuda_nodeid != (int) starpu_worker_get_memory_node(i))
					config->cuda_nodeid = -2;
				break;
			case STARPU_OPENCL_WORKER:
				if (config->opencl_nodeid == -1)
					config->opencl_nodeid = starpu_worker_get_memory_node(i);
				else if (config->opencl_nodeid != (int) starpu_worker_get_memory_node(i))
					config->opencl_nodeid = -2;
				break;
			case STARPU_MIC_WORKER:
				if (config->mic_nodeid == -1)
					config->mic_nodeid = starpu_worker_get_memory_node(i);
				else if (config->mic_nodeid != (int) starpu_worker_get_memory_node(i))
					config->mic_nodeid = -2;
				break;
			case STARPU_SCC_WORKER:
				if (config->scc_nodeid == -1)
					config->scc_nodeid = starpu_worker_get_memory_node(i);
				else if (config->scc_nodeid != (int) starpu_worker_get_memory_node(i))
					config->scc_nodeid = -2;
				break;
			case STARPU_ANY_WORKER:
				STARPU_ASSERT(0);
		}
	}

	return 0;
}

void _starpu_destroy_topology(struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_MIC
	_starpu_deinit_mp_config(config);
#endif

	/* cleanup StarPU internal data structures */
	_starpu_memory_nodes_deinit();

	_starpu_destroy_machine_config(config);
}

void
starpu_topology_print (FILE *output)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;
	unsigned pu;
	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();
	unsigned ncombinedworkers = topology->ncombinedworkers;
	unsigned nthreads_per_core = topology->nhwpus / topology->nhwcpus;

#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_t topo = topology->hwtopology;
	hwloc_obj_t pu_obj;
	hwloc_obj_t last_numa_obj = NULL, numa_obj;
	hwloc_obj_t last_package_obj = NULL, package_obj;
#endif

	for (pu = 0; pu < topology->nhwpus; pu++)
	{
#ifdef STARPU_HAVE_HWLOC
		pu_obj = hwloc_get_obj_by_type(topo, HWLOC_OBJ_PU, pu);
		numa_obj = numa_get_obj(pu_obj);
		if (numa_obj != last_numa_obj)
		{
			fprintf(output, "numa %u", numa_obj->logical_index);
			last_numa_obj = numa_obj;
		}
		fprintf(output, "\t");
		package_obj = hwloc_get_ancestor_obj_by_type(topo, HWLOC_OBJ_SOCKET, pu_obj);
		if (package_obj != last_package_obj)
		{
			fprintf(output, "pack %u", package_obj->logical_index);
			last_package_obj = package_obj;
		}
		fprintf(output, "\t");
#endif
		if ((pu % nthreads_per_core) == 0)
			fprintf(output, "core %u", pu / nthreads_per_core);
		fprintf(output, "\tPU %u\t", pu);
		for (worker = 0;
		     worker < nworkers + ncombinedworkers;
		     worker++)
		{
			if (worker < nworkers)
			{
				struct _starpu_worker *workerarg = &config->workers[worker];

				if (workerarg->bindid == (int) pu)
				{
					char name[256];
					starpu_worker_get_name (worker, name,
								sizeof(name));
					fprintf(output, "%s\t", name);
				}
			}
			else
			{
				int worker_size, i;
				int *combined_workerid;
				starpu_combined_worker_get_description(worker, &worker_size, &combined_workerid);
				for (i = 0; i < worker_size; i++)
				{
					if (topology->workers_bindid[combined_workerid[i]] == pu)
						fprintf(output, "comb %u\t", worker-nworkers);
				}
			}
		}
		fprintf(output, "\n");
	}
}
