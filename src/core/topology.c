/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012, 2013 Centre National de la Recherche Scientifique
 * Copyright (C) 2011  INRIA
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
#include <common/uthash.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#ifndef HWLOC_API_VERSION
#define HWLOC_OBJ_PU HWLOC_OBJ_PROC
#endif
#endif

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

#ifdef STARPU_SIMGRID
#include <msg/msg.h>
#include <core/simgrid.h>
#endif

static unsigned topology_is_initialized = 0;

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

static unsigned may_bind_automatically = 0;

#endif // defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL)


/*
 * Discover the topology of the machine
 */

#if defined(STARPU_USE_CUDA) || defined(STARPU_USE_OPENCL) || defined(STARPU_USE_SCC)  || defined(STARPU_SIMGRID)
static void
_starpu_initialize_workers_deviceid (int *explicit_workers_gpuid,
				  int *current, int *workers_gpuid,
				  const char *varname, unsigned nhwgpus)
{
	char *strval;
	unsigned i;

	*current = 0;

	/* conf->workers_bindid indicates the successive cpu identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cpus. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	if ((strval = getenv(varname)))
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
		 * correctly
		 * TODO: use a private value for each kind of device */
		may_bind_automatically = 1;
	}
}
#endif

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
static void
_starpu_initialize_workers_cuda_gpuid (struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = config->conf;

        _starpu_initialize_workers_deviceid (
		uconf->use_explicit_workers_cuda_gpuid == 0
		? NULL
		: (int *)uconf->workers_cuda_gpuid,
		&(config->current_cuda_gpuid),
		(int *)topology->workers_cuda_gpuid,
		"STARPU_WORKERS_CUDAID",
		topology->nhwcudagpus);
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
	struct starpu_conf *uconf = config->conf;

        _starpu_initialize_workers_deviceid(
		uconf->use_explicit_workers_opencl_gpuid == 0
		? NULL
		: (int *)uconf->workers_opencl_gpuid,
		&(config->current_opencl_gpuid),
		(int *)topology->workers_opencl_gpuid,
		"STARPU_WORKERS_OPENCLID",
		topology->nhwopenclgpus);

#ifdef STARPU_USE_CUDA
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
				entry2 = (struct handle_entry *) malloc(sizeof(*entry2));
				STARPU_ASSERT(entry2 != NULL);
				entry2->gpuid = devid;
				HASH_ADD_INT(devices_already_used, gpuid,
					     entry2);
                                tmp[nb] = devid;
                                nb ++;
                        }
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
	struct starpu_conf *uconf = config->conf;

	_starpu_initialize_workers_deviceid(
		uconf->use_explicit_workers_mic_deviceid == 0
		? NULL
		: (int *)config->user_conf->workers_mic_deviceid,
		&(config->current_mic_deviceid),
		(int *)topology->workers_mic_deviceid,
		"STARPU_WORKERS_MICID",
		topology->nhwmiccores);
}
#endif
#endif

#ifdef STARPU_USE_SCC
static void _starpu_initialize_workers_scc_deviceid(struct _starpu_machine_config *config)
{
	struct _starpu_machine_topology *topology = &config->topology;
	struct starpu_conf *uconf = config->conf;

	_starpu_initialize_workers_deviceid(
		uconf->use_explicit_workers_scc_deviceid == 0
		? NULL
		: (int *) uconf->workers_scc_deviceid,
		&(config->current_scc_deviceid),
		(int *)topology->workers_scc_deviceid,
		"STARPU_WORKERS_SCCID",
		topology->nhwscc);
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

	struct starpu_conf *user_conf = config->conf;

	char ***argv = _starpu_get_argv();
	const char *suffixes[] = {"-mic", "_mic", NULL};

	/* Environment variables to send to the Sink, it informs it what kind
	 * of node it is (architecture and type) as there is no way to discover
	 * it itself */
	char mic_idx_env[32];
	sprintf(mic_idx_env, "DEVID=%d", mic_idx);

	/* XXX: this is currently necessary so that the remote process does not
	 * segfault. */
	char nb_mic_env[32];
	sprintf(nb_mic_env, "NB_MIC=%d", 2);

	const char *mic_sink_env[] = {"STARPU_SINK=STARPU_MIC", mic_idx_env, nb_mic_env, NULL};

	char mic_sink_program_path[1024];
	/* Let's get the helper program to run on the MIC device */
	int mic_file_found =
	    _starpu_src_common_locate_file (mic_sink_program_path,
					    getenv("STARPU_MIC_SINK_PROGRAM_NAME"),
					    getenv("STARPU_MIC_SINK_PROGRAM_PATH"),
					    user_conf->mic_sink_program_path,
					    (argv ? (*argv)[0] : NULL),
					    suffixes);

	if (0 != mic_file_found) {
		fprintf(stderr, "No MIC program specified, use the environment"
			"variable STARPU_MIC_SINK_PROGRAM_NAME or the environment"
			"or the field 'starpu_conf.mic_sink_program_path'"
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

	topology->nhwcpus = 0;

#ifndef STARPU_SIMGRID
#ifdef STARPU_HAVE_HWLOC
	hwloc_topology_init(&topology->hwtopology);
	hwloc_topology_load(topology->hwtopology);
#endif
#endif

	_starpu_cpu_discover_devices(config);
	_starpu_cuda_discover_devices(config);
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

	/* conf->workers_bindid indicates the successive cpu identifier that
	 * should be used to bind the workers. It should be either filled
	 * according to the user's explicit parameters (from starpu_conf) or
	 * according to the STARPU_WORKERS_CPUID env. variable. Otherwise, a
	 * round-robin policy is used to distributed the workers over the
	 * cpus. */

	/* what do we use, explicit value, env. variable, or round-robin ? */
	if ((strval = getenv("STARPU_WORKERS_CPUID")))
	{
		/* STARPU_WORKERS_CPUID certainly contains less entries than
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
					topology->workers_bindid[i] =
						(unsigned)(val % topology->nhwcpus);
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
	else if (config->conf->use_explicit_workers_bindid)
	{
		/* we use the explicit value from the user */
		memcpy(topology->workers_bindid,
			config->conf->workers_bindid,
			STARPU_NMAXWORKERS*sizeof(unsigned));
	}
	else
	{
		/* by default, we take a round robin policy */
		for (i = 0; i < STARPU_NMAXWORKERS; i++)
			topology->workers_bindid[i] =
				(unsigned)(i % topology->nhwcpus);
	}
}

/* This function gets the identifier of the next cpu on which to bind a
 * worker. In case a list of preferred cpus was specified, we look for a an
 * available cpu among the list if possible, otherwise a round-robin policy is
 * used. */
static inline int
_starpu_get_next_bindid (struct _starpu_machine_config *config,
			 int *preferred_binding, int npreferred)
{
	struct _starpu_machine_topology *topology = &config->topology;

	unsigned found = 0;
	int current_preferred;

	for (current_preferred = 0;
	     current_preferred < npreferred;
	     current_preferred++)
	{
		if (found)
			break;

		unsigned requested_cpu = preferred_binding[current_preferred];

		/* can we bind the worker on the requested cpu ? */
		unsigned ind;
		for (ind = config->current_bindid;
		     ind < topology->nhwcpus;
		     ind++)
		{
			if (topology->workers_bindid[ind] == requested_cpu)
			{
				/* the cpu is available, we use it ! In order
				 * to make sure that it will not be used again
				 * later on, we remove the entry from the
				 * list */
				topology->workers_bindid[ind] =
					topology->workers_bindid[config->current_bindid];
				topology->workers_bindid[config->current_bindid] = requested_cpu;

				found = 1;

				break;
			}
		}
	}

	unsigned i = ((config->current_bindid++) % STARPU_NMAXWORKERS);

	return (int)topology->workers_bindid[i];
}

unsigned
_starpu_topology_get_nhwcpu (struct _starpu_machine_config *config)
{
#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	_starpu_opencl_init();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	return config->topology.nhwcpus;
}

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
	nmiccores = starpu_get_env_number("STARPU_NMIC");

	/* STARPU_NMIC is not set. Did the user specify anything ? */
	if (nmiccores == -1 && user_conf)
		nmiccores = user_conf->nmic;

	if (nmiccores != 0)
	{
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
				/* The user requires more MIC devices than there is available */
				fprintf(stderr,
					"# Warning: %d MIC devices requested. Only %d available.\n",
					nmiccores, topology->nhwmiccores[mic_idx]);
				nmiccores = topology->nhwmiccores[mic_idx];
			}
		}
	}

	topology->nmiccores[mic_idx] = nmiccores;
	STARPU_ASSERT(topology->nmiccores[mic_idx] + topology->nworkers <= STARPU_NMAXWORKERS);

	/* _starpu_initialize_workers_mic_deviceid (config); */

	unsigned miccore_id;
	for (miccore_id = 0; miccore_id < topology->nmiccores[mic_idx]; miccore_id++)
	{
		int worker_idx = topology->nworkers + miccore_id;
		enum starpu_perfmodel_archtype arch =
			(enum starpu_perfmodel_archtype)((int)STARPU_MIC_DEFAULT + devid);
		config->workers[worker_idx].arch = STARPU_MIC_WORKER;
		config->workers[worker_idx].perf_arch = arch;
		config->workers[worker_idx].mp_nodeid = mic_idx;
		config->workers[worker_idx].devid = miccore_id;
		config->workers[worker_idx].worker_mask = STARPU_MIC;
		config->worker_mask |= STARPU_MIC;
		_starpu_init_sched_ctx_for_worker(config->workers[worker_idx].workerid);
	}

	topology->nworkers += topology->nmiccores[mic_idx];
    }


#ifdef STARPU_USE_MIC
static COIENGINE handles[2];
static COIPROCESS process[2];
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

	int reqmicdevices = starpu_get_env_number("STARPU_NMICDEVS");
	if (-1 == reqmicdevices)
		reqmicdevices = nhwmicdevices;

	topology->nmicdevices = 0;
	unsigned i;
	for (i = 0; i < STARPU_MIN (nhwmicdevices, (unsigned) reqmicdevices); i++)
		if (0 == _starpu_init_mic_node (config, i, &handles[i], &process[i]))
			topology->nmicdevices++;

	i = 0;
	for (; i < topology->nmicdevices; i++)
		_starpu_init_mic_config (config, user_conf, i);
#endif
}

static void
_starpu_deinit_mic_node (unsigned mic_idx)
{
	_starpu_mp_common_send_command(mic_nodes[mic_idx], STARPU_EXIT, NULL, 0);

	COIProcessDestroy(process[mic_idx], -1, 0, NULL, NULL);

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
_starpu_init_machine_config (struct _starpu_machine_config *config, int no_mp_config)
{
	int i;
	for (i = 0; i < STARPU_NMAXWORKERS; i++)
		config->workers[i].workerid = i;

	struct _starpu_machine_topology *topology = &config->topology;

	topology->nworkers = 0;
	topology->ncombinedworkers = 0;
	topology->nsched_ctxs = 0;

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	_starpu_opencl_init();
#endif
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	_starpu_initialize_workers_bindid(config);

#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
	int ncuda = config->conf->ncuda;

	if (ncuda != 0)
	{
		/* The user did not disable CUDA. We need to initialize CUDA
 		 * early to count the number of devices */
		_starpu_init_cuda();
		int nb_devices = _starpu_get_cuda_device_count();

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
	STARPU_ASSERT(topology->ncudagpus <= STARPU_MAXCUDADEVS);

	_starpu_initialize_workers_cuda_gpuid(config);

	unsigned cudagpu;
	for (cudagpu = 0; cudagpu < topology->ncudagpus; cudagpu++)
	{
		int worker_idx = topology->nworkers + cudagpu;
		config->workers[worker_idx].arch = STARPU_CUDA_WORKER;
		int devid = _starpu_get_next_cuda_gpuid(config);
		enum starpu_perfmodel_archtype arch =
			(enum starpu_perfmodel_archtype)((int)STARPU_CUDA_DEFAULT + devid);
		config->workers[worker_idx].mp_nodeid = -1;
		config->workers[worker_idx].devid = devid;
		config->workers[worker_idx].perf_arch = arch;
		config->workers[worker_idx].worker_mask = STARPU_CUDA;
		_starpu_init_sched_ctx_for_worker(config->workers[worker_idx].workerid);
		config->worker_mask |= STARPU_CUDA;

		struct handle_entry *entry;
		entry = (struct handle_entry *) malloc(sizeof(*entry));
		STARPU_ASSERT(entry != NULL);
		entry->gpuid = devid;
		HASH_ADD_INT(devices_using_cuda, gpuid, entry);
        }

	topology->nworkers += topology->ncudagpus;
#endif

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
	int nopencl = config->conf->nopencl;

	if (nopencl != 0)
	{
		/* The user did not disable OPENCL. We need to initialize
 		 * OpenCL early to count the number of devices */
		_starpu_opencl_init();
		int nb_devices;
		nb_devices = _starpu_opencl_get_device_count();

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
		{ // There is no more devices left
			topology->nopenclgpus = openclgpu;
			break;
		}
		config->workers[worker_idx].arch = STARPU_OPENCL_WORKER;
		enum starpu_perfmodel_archtype arch =
			(enum starpu_perfmodel_archtype)((int)STARPU_OPENCL_DEFAULT + devid);
		config->workers[worker_idx].mp_nodeid = -1;
		config->workers[worker_idx].devid = devid;
		config->workers[worker_idx].perf_arch = arch;
		config->workers[worker_idx].worker_mask = STARPU_OPENCL;
		_starpu_init_sched_ctx_for_worker(config->workers[worker_idx].workerid);
		config->worker_mask |= STARPU_OPENCL;
	}

	topology->nworkers += topology->nopenclgpus;
#endif

#ifdef STARPU_USE_SCC
	int nscc = config->conf->nscc;

	unsigned nb_scc_nodes = _starpu_scc_src_get_device_count();

	if (nscc != 0)
	{
		/* The user did not disable SCC. We need to count
		 * the number of devices */
		int nb_devices = nb_scc_nodes;

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
		enum starpu_perfmodel_archtype arch = (enum starpu_perfmodel_archtype)((int)STARPU_SCC_DEFAULT + devid);
		config->workers[topology->nworkers + sccdev].mp_nodeid = -1;
		config->workers[topology->nworkers + sccdev].devid = devid;
		config->workers[topology->nworkers + sccdev].perf_arch = arch;
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
	    _starpu_init_mp_config (config, config->conf);
#endif

/* we put the CPU section after the accelerator : in case there was an
 * accelerator found, we devote one cpu */
#if defined(STARPU_USE_CPU) || defined(STARPU_SIMGRID)
	int ncpu = config->conf->ncpus;

	if (ncpu != 0)
	{
		if (ncpu == -1)
		{
			unsigned mic_busy_cpus = 0;
			unsigned i = 0;
			for (i = 0; i < STARPU_MAXMICDEVS; i++)
				mic_busy_cpus += (topology->nmiccores[i] ? 1 : 0);

			unsigned already_busy_cpus = mic_busy_cpus + topology->ncudagpus
				+ topology->nopenclgpus + topology->nsccdevices;

			long avail_cpus = (long) topology->nhwcpus - (long) already_busy_cpus;
			if (avail_cpus < 0)
				avail_cpus = 0;
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
	for (cpu = 0; cpu < topology->ncpus; cpu++)
	{
		int worker_idx = topology->nworkers + cpu;
		config->workers[worker_idx].arch = STARPU_CPU_WORKER;
		config->workers[worker_idx].perf_arch = STARPU_CPU_DEFAULT;
		config->workers[worker_idx].mp_nodeid = -1;
		config->workers[worker_idx].devid = cpu;
		config->workers[worker_idx].worker_mask = STARPU_CPU;
		config->worker_mask |= STARPU_CPU;
		_starpu_init_sched_ctx_for_worker(config->workers[worker_idx].workerid);
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



void
_starpu_bind_thread_on_cpu (
	struct _starpu_machine_config *config STARPU_ATTRIBUTE_UNUSED,
	unsigned cpuid)
{
#ifdef STARPU_SIMGRID
	return;
#endif
	if (starpu_get_env_number("STARPU_WORKERS_NOBIND") > 0)
		return;
#ifdef STARPU_HAVE_HWLOC
	const struct hwloc_topology_support *support;

#ifdef STARPU_USE_OPENCL
	_starpu_opencl_init();
#endif
#ifdef STARPU_USE_CUDA
	_starpu_init_cuda();
#endif
	_starpu_init_topology(config);

	support = hwloc_topology_get_support (config->topology.hwtopology);
	if (support->cpubind->set_thisthread_cpubind)
	{
		hwloc_obj_t obj =
			hwloc_get_obj_by_depth (config->topology.hwtopology,
						config->cpu_depth, cpuid);
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
		perror("binding thread");
		STARPU_ABORT();
	}

#elif defined(__MINGW32__) || defined(__CYGWIN__)
	DWORD mask = 1 << cpuid;
	if (!SetThreadAffinityMask(GetCurrentThread(), mask))
	{
		_STARPU_ERROR("SetThreadMaskAffinity(%lx) failed\n", mask);
	}
#else
#warning no CPU binding support
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
	_starpu_opencl_init();
#endif
#ifdef STARPU_USE_CUDA
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
#warning no parallel worker CPU binding support
#endif
}


static void
_starpu_init_workers_binding (struct _starpu_machine_config *config, int no_mp_config)
{
	/* launch one thread per CPU */
	unsigned ram_memory_node;

	/* a single cpu is dedicated for the accelerators */
	int accelerator_bindid = -1;

	/* note that even if the CPU cpu are not used, we always have a RAM
	 * node */
	/* TODO : support NUMA  ;) */
	ram_memory_node = _starpu_memory_node_register(STARPU_CPU_RAM, -1);

#ifdef STARPU_SIMGRID
	char name[16];
	xbt_dynar_t hosts = MSG_hosts_as_dynar();
	msg_host_t host = MSG_get_host_by_name("RAM");
	STARPU_ASSERT(host);
	_starpu_simgrid_memory_node_set_host(0, host);
#endif

	/* We will store all the busid of the different (src, dst)
	 * combinations in a matrix which we initialize here. */
	_starpu_initialize_busid_matrix();

#ifdef STARPU_USE_MIC
	/* Each MIC device has its own memory node. */
	unsigned mic_memory_nodes[STARPU_MAXMICDEVS];

	// Register the memory nodes for the MIC devices.
	if (! no_mp_config) {
	    unsigned i = 0;
	    for (i = 0; i < config->topology.nmicdevices; i++) {
		mic_memory_nodes[i] = _starpu_memory_node_register (STARPU_MIC_RAM, i);
		_starpu_register_bus(0, mic_memory_nodes[i]);
		_starpu_register_bus(mic_memory_nodes[i], 0);
	    }
	}
#endif

	unsigned worker;
	for (worker = 0; worker < config->topology.nworkers; worker++)
	{
		unsigned memory_node = -1;
		unsigned is_a_set_of_accelerators = 0;
		struct _starpu_worker *workerarg = &config->workers[worker];

		/* Perhaps the worker has some "favourite" bindings  */
		int *preferred_binding = NULL;
		int npreferred = 0;

		/* select the memory node that contains worker's memory */
		switch (workerarg->arch)
		{
			case STARPU_CPU_WORKER:
			/* "dedicate" a cpu cpu to that worker */
				is_a_set_of_accelerators = 0;
				memory_node = ram_memory_node;
				_starpu_memory_node_add_nworkers(ram_memory_node);
				break;
#if defined(STARPU_USE_CUDA) || defined(STARPU_SIMGRID)
			case STARPU_CUDA_WORKER:
#ifndef STARPU_SIMGRID
				if (may_bind_automatically)
				{
					/* StarPU is allowed to bind threads automatically */
					preferred_binding = _starpu_get_cuda_affinity_vector(workerarg->devid);
					npreferred = config->topology.nhwcpus;
				}
#endif
				is_a_set_of_accelerators = 0;
				memory_node = _starpu_memory_node_register(STARPU_CUDA_RAM, workerarg->devid);
#ifdef STARPU_SIMGRID
				snprintf(name, sizeof(name), "CUDA%d", workerarg->devid);
				host = MSG_get_host_by_name(name);
				STARPU_ASSERT(host);
				_starpu_simgrid_memory_node_set_host(memory_node, host);
#endif
				_starpu_memory_node_add_nworkers(memory_node);

				_starpu_register_bus(0, memory_node);
				_starpu_register_bus(memory_node, 0);
#ifdef HAVE_CUDA_MEMCPY_PEER
				unsigned worker2;
				for (worker2 = 0; worker2 < worker; worker2++)
				{
					struct _starpu_worker *workerarg = &config->workers[worker];
					if (workerarg->arch == STARPU_CUDA_WORKER)
					{
						unsigned memory_node2 = starpu_worker_get_memory_node(worker2);
						_starpu_register_bus(memory_node2, memory_node);
						_starpu_register_bus(memory_node, memory_node2);
					}
				}
#endif
				break;
#endif

#if defined(STARPU_USE_OPENCL) || defined(STARPU_SIMGRID)
		        case STARPU_OPENCL_WORKER:
#ifndef STARPU_SIMGRID
				if (may_bind_automatically)
				{
					/* StarPU is allowed to bind threads automatically */
					preferred_binding = _starpu_get_opencl_affinity_vector(workerarg->devid);
					npreferred = config->topology.nhwcpus;
				}
#endif
				is_a_set_of_accelerators = 0;
				memory_node = _starpu_memory_node_register(STARPU_OPENCL_RAM, workerarg->devid);
#ifdef STARPU_SIMGRID
				snprintf(name, sizeof(name), "OpenCL%d", workerarg->devid);
				host = MSG_get_host_by_name(name);
				STARPU_ASSERT(host);
				_starpu_simgrid_memory_node_set_host(memory_node, host);
#endif
				_starpu_memory_node_add_nworkers(memory_node);
				_starpu_register_bus(0, memory_node);
				_starpu_register_bus(memory_node, 0);
				break;
#endif

#ifdef STARPU_USE_MIC
		        case STARPU_MIC_WORKER:
				//if (may_bind_automatically)
				//{
				//	/* StarPU is allowed to bind threads automatically */
				//	preferred_binding = _starpu_get_mic_affinity_vector(workerarg->devid);
				//	npreferred = config->topology.nhwcpus;
				//}
				is_a_set_of_accelerators = 1;
				memory_node = mic_memory_nodes[workerarg->mp_nodeid];
				_starpu_memory_node_add_nworkers(memory_node);
				/* memory_node = _starpu_memory_node_register(STARPU_MIC_RAM, workerarg->devid);*/

				/* _starpu_register_bus(0, memory_node);
				 * _starpu_register_bus(memory_node, 0); */
				break;
#endif /* STARPU_USE_MIC */

#ifdef STARPU_USE_SCC
			case STARPU_SCC_WORKER:
			{
				/* Node 0 represents the SCC shared memory when we're on SCC. */
				struct _starpu_memory_node_descr *descr = _starpu_memory_node_get_description();
				descr->nodes[ram_memory_node] = STARPU_SCC_SHM;

				is_a_set_of_accelerators = 0;
				memory_node = ram_memory_node;
				_starpu_memory_node_add_nworkers(memory_node);
			}
				break;
#endif

			default:
				STARPU_ABORT();
		}

		if (is_a_set_of_accelerators)
		{
			if (accelerator_bindid == -1)
				accelerator_bindid = _starpu_get_next_bindid(config, preferred_binding, npreferred);

			workerarg->bindid = accelerator_bindid;
		}
		else
		{
			workerarg->bindid = _starpu_get_next_bindid(config, preferred_binding, npreferred);
		}

		workerarg->memory_node = memory_node;

#ifdef __GLIBC__
		/* Save the initial cpuset */
		CPU_ZERO(&workerarg->cpu_set);
		CPU_SET(workerarg->bindid, &workerarg->cpu_set);
#endif /* __GLIBC__ */

#ifdef STARPU_HAVE_HWLOC
		/* Put the worker descriptor in the userdata field of the
		 * hwloc object describing the CPU */
		hwloc_obj_t worker_obj;
		worker_obj =
			hwloc_get_obj_by_depth (config->topology.hwtopology,
						config->cpu_depth,
						workerarg->bindid);
		worker_obj->userdata = &config->workers[worker];

		/* Clear the cpu set and set the cpu */
		workerarg->hwloc_cpu_set =
			hwloc_bitmap_dup (worker_obj->cpuset);
#endif
	}
#ifdef STARPU_SIMGRID
	xbt_dynar_free(&hosts);
#endif
}


int
_starpu_build_topology (struct _starpu_machine_config *config, int no_mp_config)
{
	int ret;

	ret = _starpu_init_machine_config(config, no_mp_config);
	if (ret)
		return ret;

	/* for the data management library */
	_starpu_memory_nodes_init();

	_starpu_init_workers_binding(config, no_mp_config);

	return 0;
}

void
_starpu_destroy_topology (
	struct _starpu_machine_config *config __attribute__ ((unused)))
{
#ifdef STARPU_USE_MIC
	_starpu_deinit_mp_config(config);
#endif

	/* cleanup StarPU internal data structures */
	_starpu_memory_nodes_deinit();

	unsigned worker;
	for (worker = 0; worker < config->topology.nworkers; worker++)
	{
#ifdef STARPU_HAVE_HWLOC
		struct _starpu_worker *workerarg = &config->workers[worker];
		hwloc_bitmap_free(workerarg->hwloc_cpu_set);
#endif
	}

#ifdef STARPU_HAVE_HWLOC
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
	may_bind_automatically = 0;
#endif
}

void
starpu_topology_print (FILE *output)
{
	struct _starpu_machine_config *config = _starpu_get_machine_config();
	struct _starpu_machine_topology *topology = &config->topology;
	unsigned core;
	unsigned worker;
	unsigned nworkers = starpu_worker_get_count();
	unsigned ncombinedworkers = topology->ncombinedworkers;

	for (core = 0; core < topology->nhwcpus; core++)
	{
		fprintf(output, "core %u\t", core);
		for (worker = 0;
		     worker < nworkers + ncombinedworkers;
		     worker++)
		{
			if (worker < nworkers)
			{
				if (topology->workers_bindid[worker] == core)
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
					if (topology->workers_bindid[combined_workerid[i]] == core)
						fprintf(output, "comb %u\t", worker-nworkers);
				}
			}
		}
		fprintf(output, "\n");
	}
}
