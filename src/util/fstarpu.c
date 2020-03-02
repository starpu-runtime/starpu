/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2016-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <string.h>
#include <starpu.h>
#include <common/utils.h>

typedef void (*_starpu_callback_func_t)(void *);

static const intptr_t fstarpu_r	= STARPU_R;
static const intptr_t fstarpu_w	= STARPU_W;
static const intptr_t fstarpu_rw	= STARPU_RW;
static const intptr_t fstarpu_scratch	= STARPU_SCRATCH;
static const intptr_t fstarpu_redux	= STARPU_REDUX;
static const intptr_t fstarpu_commute	= STARPU_COMMUTE;
static const intptr_t fstarpu_ssend	= STARPU_SSEND;
static const intptr_t fstarpu_locality	= STARPU_LOCALITY;

static const intptr_t fstarpu_data_array	= STARPU_DATA_ARRAY;
static const intptr_t fstarpu_data_mode_array	= STARPU_DATA_MODE_ARRAY;
static const intptr_t fstarpu_cl_args	= STARPU_CL_ARGS;
static const intptr_t fstarpu_cl_args_nfree	= STARPU_CL_ARGS_NFREE;
static const intptr_t fstarpu_task_deps_array	= STARPU_TASK_DEPS_ARRAY;
static const intptr_t fstarpu_task_end_deps_array	= STARPU_TASK_END_DEPS_ARRAY;
static const intptr_t fstarpu_callback	= STARPU_CALLBACK;
static const intptr_t fstarpu_callback_with_arg	= STARPU_CALLBACK_WITH_ARG;
static const intptr_t fstarpu_callback_with_arg_nfree	= STARPU_CALLBACK_WITH_ARG_NFREE;
static const intptr_t fstarpu_callback_arg	= STARPU_CALLBACK_ARG;
static const intptr_t fstarpu_callback_arg_nfree= STARPU_CALLBACK_ARG_NFREE;
static const intptr_t fstarpu_prologue_callback	= STARPU_PROLOGUE_CALLBACK;
static const intptr_t fstarpu_prologue_callback_arg	  = STARPU_PROLOGUE_CALLBACK_ARG;
static const intptr_t fstarpu_prologue_callback_arg_nfree = STARPU_PROLOGUE_CALLBACK_ARG_NFREE;
static const intptr_t fstarpu_prologue_callback_pop	  	= STARPU_PROLOGUE_CALLBACK_POP;
static const intptr_t fstarpu_prologue_callback_pop_arg	  	= STARPU_PROLOGUE_CALLBACK_POP_ARG;
static const intptr_t fstarpu_prologue_callback_pop_arg_nfree	 = STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE;
static const intptr_t fstarpu_priority	= STARPU_PRIORITY;
static const intptr_t fstarpu_execute_on_node	= STARPU_EXECUTE_ON_NODE;
static const intptr_t fstarpu_execute_on_data	= STARPU_EXECUTE_ON_DATA;
static const intptr_t fstarpu_execute_where	= STARPU_EXECUTE_WHERE;
static const intptr_t fstarpu_execute_on_worker	= STARPU_EXECUTE_ON_WORKER;
static const intptr_t fstarpu_worker_order	= STARPU_WORKER_ORDER;
static const intptr_t fstarpu_hypervisor_tag	= STARPU_HYPERVISOR_TAG;
static const intptr_t fstarpu_possibly_parallel	= STARPU_POSSIBLY_PARALLEL;
static const intptr_t fstarpu_flops	= STARPU_FLOPS;
static const intptr_t fstarpu_tag	= STARPU_TAG;
static const intptr_t fstarpu_tag_only	= STARPU_TAG_ONLY;
static const intptr_t fstarpu_name	= STARPU_NAME;
static const intptr_t fstarpu_task_color	= STARPU_TASK_COLOR;
static const intptr_t fstarpu_handles_sequential_consistency	= STARPU_HANDLES_SEQUENTIAL_CONSISTENCY;
static const intptr_t fstarpu_task_end_dep	= STARPU_TASK_END_DEP;
static const intptr_t fstarpu_task_synchronous	= STARPU_TASK_SYNCHRONOUS;
static const intptr_t fstarpu_node_selection_policy	= STARPU_NODE_SELECTION_POLICY;
static const intptr_t fstarpu_task_workerids = STARPU_TASK_WORKERIDS;
static const intptr_t fstarpu_sequential_consistency = STARPU_SEQUENTIAL_CONSISTENCY;
static const intptr_t fstarpu_task_profiling_info = STARPU_TASK_PROFILING_INFO;
static const intptr_t fstarpu_task_no_submitorder = STARPU_TASK_NO_SUBMITORDER;
static const intptr_t fstarpu_task_sched_data = STARPU_TASK_SCHED_DATA;

static const intptr_t fstarpu_value = STARPU_VALUE;
static const intptr_t fstarpu_sched_ctx = STARPU_SCHED_CTX;

static const intptr_t fstarpu_cpu_worker = STARPU_CPU_WORKER;
static const intptr_t fstarpu_cuda_worker = STARPU_CUDA_WORKER;
static const intptr_t fstarpu_opencl_worker = STARPU_OPENCL_WORKER;
static const intptr_t fstarpu_mic_worker = STARPU_MIC_WORKER;
static const intptr_t fstarpu_any_worker = STARPU_ANY_WORKER;

static const intptr_t fstarpu_nmaxbufs = STARPU_NMAXBUFS;

static const intptr_t fstarpu_sched_ctx_policy_name	= STARPU_SCHED_CTX_POLICY_NAME;
static const intptr_t fstarpu_sched_ctx_policy_struct	= STARPU_SCHED_CTX_POLICY_STRUCT;
static const intptr_t fstarpu_sched_ctx_policy_min_prio	= STARPU_SCHED_CTX_POLICY_MIN_PRIO;
static const intptr_t fstarpu_sched_ctx_policy_max_prio	= STARPU_SCHED_CTX_POLICY_MAX_PRIO;
static const intptr_t fstarpu_sched_ctx_hierarchy_level	= STARPU_SCHED_CTX_HIERARCHY_LEVEL;
static const intptr_t fstarpu_sched_ctx_nested	= STARPU_SCHED_CTX_NESTED;
static const intptr_t fstarpu_sched_ctx_awake_workers	= STARPU_SCHED_CTX_AWAKE_WORKERS;
static const intptr_t fstarpu_sched_ctx_policy_init	= STARPU_SCHED_CTX_POLICY_INIT;
static const intptr_t fstarpu_sched_ctx_user_data	= STARPU_SCHED_CTX_USER_DATA;

static const intptr_t fstarpu_starpu_nowhere	= STARPU_NOWHERE;
static const intptr_t fstarpu_starpu_cpu	= STARPU_CPU;
static const intptr_t fstarpu_starpu_cuda	= STARPU_CUDA;
static const intptr_t fstarpu_starpu_opencl	= STARPU_OPENCL;
static const intptr_t fstarpu_starpu_mic	= STARPU_MIC;

static const intptr_t fstarpu_starpu_codelet_simgrid_execute	= STARPU_CODELET_SIMGRID_EXECUTE;
static const intptr_t fstarpu_starpu_codelet_simgrid_execute_and_inject	= STARPU_CODELET_SIMGRID_EXECUTE_AND_INJECT;
static const intptr_t fstarpu_starpu_cuda_async	= STARPU_CUDA_ASYNC;
static const intptr_t fstarpu_starpu_opencl_async	= STARPU_OPENCL_ASYNC;


intptr_t fstarpu_get_constant(char *s)
{
	if	(!strcmp(s, "FSTARPU_R"))	{ return fstarpu_r; }
	else if	(!strcmp(s, "FSTARPU_W"))	{ return fstarpu_w; }
	else if	(!strcmp(s, "FSTARPU_RW"))	{ return fstarpu_rw; }
	else if	(!strcmp(s, "FSTARPU_SCRATCH"))	{ return fstarpu_scratch; }
	else if	(!strcmp(s, "FSTARPU_REDUX"))	{ return fstarpu_redux; }
	else if	(!strcmp(s, "FSTARPU_COMMUTE"))	{ return fstarpu_commute; }
	else if	(!strcmp(s, "FSTARPU_SSEND"))	{ return fstarpu_ssend; }
	else if	(!strcmp(s, "FSTARPU_LOCALITY"))	{ return fstarpu_locality; }


	else if	(!strcmp(s, "FSTARPU_DATA_ARRAY"))	{ return fstarpu_data_array; }
	else if	(!strcmp(s, "FSTARPU_DATA_MODE_ARRAY"))	{ return fstarpu_data_mode_array; }
	else if	(!strcmp(s, "FSTARPU_CL_ARGS"))	{ return fstarpu_cl_args; }
	else if	(!strcmp(s, "FSTARPU_CL_ARGS_NFREE"))	{ return fstarpu_cl_args_nfree; }
	else if	(!strcmp(s, "FSTARPU_TASK_DEPS_ARRAY"))	{ return fstarpu_task_deps_array; }
	else if	(!strcmp(s, "FSTARPU_TASK_END_DEPS_ARRAY"))	{ return fstarpu_task_end_deps_array; }
	else if	(!strcmp(s, "FSTARPU_CALLBACK"))	{ return fstarpu_callback; }
	else if	(!strcmp(s, "FSTARPU_CALLBACK_WITH_ARG"))	{ return fstarpu_callback_with_arg; }
	else if	(!strcmp(s, "FSTARPU_CALLBACK_WITH_ARG_NFREE"))	{ return fstarpu_callback_with_arg_nfree; }
	else if	(!strcmp(s, "FSTARPU_CALLBACK_ARG"))		{ return fstarpu_callback_arg; }
	else if	(!strcmp(s, "FSTARPU_CALLBACK_ARG_NFREE"))	{ return fstarpu_callback_arg_nfree; }
	else if	(!strcmp(s, "FSTARPU_PROLOGUE_CALLBACK"))	{ return fstarpu_prologue_callback; }
	else if	(!strcmp(s, "FSTARPU_PROLOGUE_CALLBACK_ARG"))	{ return fstarpu_prologue_callback_arg; }
	else if	(!strcmp(s, "FSTARPU_PROLOGUE_CALLBACK_ARG_NFREE"))	{ return fstarpu_prologue_callback_arg_nfree; }
	else if	(!strcmp(s, "FSTARPU_PROLOGUE_CALLBACK_POP"))	{ return fstarpu_prologue_callback_pop; }
	else if	(!strcmp(s, "FSTARPU_PROLOGUE_CALLBACK_POP_ARG"))	{ return fstarpu_prologue_callback_pop_arg; }
	else if	(!strcmp(s, "FSTARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE"))	{ return fstarpu_prologue_callback_pop_arg_nfree; }
	else if	(!strcmp(s, "FSTARPU_PRIORITY"))	{ return fstarpu_priority; }
	else if	(!strcmp(s, "FSTARPU_EXECUTE_ON_NODE"))	{ return fstarpu_execute_on_node; }
	else if	(!strcmp(s, "FSTARPU_EXECUTE_ON_DATA"))	{ return fstarpu_execute_on_data; }
	else if	(!strcmp(s, "FSTARPU_EXECUTE_WHERE"))	{ return fstarpu_execute_where; }
	else if	(!strcmp(s, "FSTARPU_EXECUTE_ON_WORKER"))	{ return fstarpu_execute_on_worker; }
	else if	(!strcmp(s, "FSTARPU_WORKER_ORDER"))	{ return fstarpu_worker_order; }
	else if	(!strcmp(s, "FSTARPU_HYPERVISOR_TAG"))	{ return fstarpu_hypervisor_tag; }
	else if	(!strcmp(s, "FSTARPU_POSSIBLY_PARALLEL"))	{ return fstarpu_possibly_parallel; }
	else if	(!strcmp(s, "FSTARPU_FLOPS"))	{ return fstarpu_flops; }
	else if	(!strcmp(s, "FSTARPU_TAG"))	{ return fstarpu_tag; }
	else if	(!strcmp(s, "FSTARPU_TAG_ONLY"))	{ return fstarpu_tag_only; }
	else if	(!strcmp(s, "FSTARPU_NAME"))	{ return fstarpu_name; }
	else if	(!strcmp(s, "FSTARPU_NODE_SELECTION_POLICY"))	{ return fstarpu_node_selection_policy; }
	else if (!strcmp(s, "FSTARPU_VALUE"))	{ return fstarpu_value; }
	else if (!strcmp(s, "FSTARPU_SCHED_CTX"))	{ return fstarpu_sched_ctx; }
	else if (!strcmp(s, "FSTARPU_TASK_COLOR"))	{ return fstarpu_task_color; }
	else if (!strcmp(s, "FSTARPU_HANDLES_SEQUENTIAL_CONSISTENCY"))	{ return fstarpu_handles_sequential_consistency; }
	else if (!strcmp(s, "FSTARPU_TASK_END_DEP"))	{ return fstarpu_task_end_dep; }
	else if (!strcmp(s, "FSTARPU_TASK_WORKERIDS"))	{ return fstarpu_task_workerids; }
	else if (!strcmp(s, "FSTARPU_TASK_SYNCHRONOUS"))	{ return fstarpu_task_synchronous; }
	else if (!strcmp(s, "FSTARPU_SEQUENTIAL_CONSISTENCY"))	{ return fstarpu_sequential_consistency; }
	else if (!strcmp(s, "FSTARPU_TASK_PROFILING_INFO"))	{ return fstarpu_task_profiling_info; }
	else if (!strcmp(s, "FSTARPU_TASK_NO_SUBMITORDER"))	{ return fstarpu_task_no_submitorder; }
	else if	(!strcmp(s, "FSTARPU_TASK_SCHED_DATA"))	{ return fstarpu_task_sched_data; }

	else if (!strcmp(s, "FSTARPU_CPU_WORKER"))	{ return fstarpu_cpu_worker; }
	else if (!strcmp(s, "FSTARPU_CUDA_WORKER"))	{ return fstarpu_cuda_worker; }
	else if (!strcmp(s, "FSTARPU_OPENCL_WORKER"))	{ return fstarpu_opencl_worker; }
	else if (!strcmp(s, "FSTARPU_MIC_WORKER"))	{ return fstarpu_mic_worker; }
	else if (!strcmp(s, "FSTARPU_ANY_WORKER"))	{ return fstarpu_any_worker; }

	else if (!strcmp(s, "FSTARPU_NMAXBUFS"))	{ return fstarpu_nmaxbufs; }

	else if (!strcmp(s, "FSTARPU_SCHED_CTX_POLICY_NAME"))	{ return fstarpu_sched_ctx_policy_name; }
	else if (!strcmp(s, "FSTARPU_SCHED_CTX_POLICY_STRUCT"))	{ return fstarpu_sched_ctx_policy_struct; }
	else if (!strcmp(s, "FSTARPU_SCHED_CTX_POLICY_MIN_PRIO"))	{ return fstarpu_sched_ctx_policy_min_prio; }
	else if (!strcmp(s, "FSTARPU_SCHED_CTX_POLICY_MAX_PRIO"))	{ return fstarpu_sched_ctx_policy_max_prio; }
	else if (!strcmp(s, "FSTARPU_SCHED_CTX_HIERARCHY_LEVEL"))	{ return fstarpu_sched_ctx_hierarchy_level; }
	else if (!strcmp(s, "FSTARPU_SCHED_CTX_NESTED"))	{ return fstarpu_sched_ctx_nested; }
	else if (!strcmp(s, "FSTARPU_SCHED_CTX_AWAKE_WORKERS"))	{ return fstarpu_sched_ctx_awake_workers; }
	else if (!strcmp(s, "FSTARPU_SCHED_CTX_POLICY_INIT"))	{ return fstarpu_sched_ctx_policy_init; }
	else if (!strcmp(s, "FSTARPU_SCHED_CTX_USER_DATA"))	{ return fstarpu_sched_ctx_user_data; }

	else if (!strcmp(s, "FSTARPU_NOWHERE"))	{ return fstarpu_starpu_nowhere; }
	else if (!strcmp(s, "FSTARPU_CPU"))	{ return fstarpu_starpu_cpu; }
	else if (!strcmp(s, "FSTARPU_CUDA"))	{ return fstarpu_starpu_cuda; }
	else if (!strcmp(s, "FSTARPU_OPENCL"))	{ return fstarpu_starpu_opencl; }
	else if (!strcmp(s, "FSTARPU_MIC"))	{ return fstarpu_starpu_mic; }

	else if (!strcmp(s, "FSTARPU_CODELET_SIMGRID_EXECUTE"))	{ return fstarpu_starpu_codelet_simgrid_execute; }
	else if (!strcmp(s, "FSTARPU_CODELET_SIMGRID_EXECUTE_AND_INJECT"))	{ return fstarpu_starpu_codelet_simgrid_execute_and_inject; }
	else if (!strcmp(s, "FSTARPU_CUDA_ASYNC"))	{ return fstarpu_starpu_cuda_async; }
	else if (!strcmp(s, "FSTARPU_OPENCL_ASYNC"))	{ return fstarpu_starpu_opencl_async; }

	else { _STARPU_ERROR("unknown constant"); }
}

struct starpu_conf *fstarpu_conf_allocate(void)
{
	struct starpu_conf *conf;
	_STARPU_MALLOC(conf, sizeof(*conf));
	starpu_conf_init(conf);
	return conf;
}

void fstarpu_conf_free(struct starpu_conf *conf)
{
	memset(conf, 0, sizeof(*conf));
	free(conf);
}

void fstarpu_conf_set_sched_policy_name(struct starpu_conf *conf, const char *sched_policy_name)
{
	conf->sched_policy_name = sched_policy_name;
}

void fstarpu_conf_set_min_prio(struct starpu_conf *conf, int min_prio)
{
	conf->global_sched_ctx_min_priority = min_prio;
}

void fstarpu_conf_set_max_prio(struct starpu_conf *conf, int max_prio)
{
	conf->global_sched_ctx_max_priority = max_prio;
}

void fstarpu_conf_set_ncpu(struct starpu_conf *conf, int ncpu)
{
	STARPU_ASSERT(ncpu >= 0 && ncpu <= STARPU_NMAXWORKERS);
	conf->ncpus = ncpu;
}

void fstarpu_conf_set_ncuda(struct starpu_conf *conf, int ncuda)
{
	STARPU_ASSERT(ncuda >= 0 && ncuda <= STARPU_NMAXWORKERS);
	conf->ncuda = ncuda;
}

void fstarpu_conf_set_nopencl(struct starpu_conf *conf, int nopencl)
{
	STARPU_ASSERT(nopencl >= 0 && nopencl <= STARPU_NMAXWORKERS);
	conf->nopencl = nopencl;
}

void fstarpu_conf_set_nmic(struct starpu_conf *conf, int nmic)
{
	STARPU_ASSERT(nmic >= 0 && nmic <= STARPU_NMAXWORKERS);
	conf->nmic = nmic;
}

void fstarpu_conf_set_calibrate(struct starpu_conf *conf, int calibrate)
{
	STARPU_ASSERT(calibrate == 0 || calibrate == 1);
	conf->calibrate = calibrate;
}

void fstarpu_conf_set_bus_calibrate(struct starpu_conf *conf, int bus_calibrate)
{
	STARPU_ASSERT(bus_calibrate == 0 || bus_calibrate == 1);
	conf->bus_calibrate = bus_calibrate;
}

void fstarpu_topology_print(void)
{
	starpu_topology_print(stderr);
}

struct starpu_codelet *fstarpu_codelet_allocate(void)
{
	struct starpu_codelet *cl;
	_STARPU_MALLOC(cl, sizeof(*cl));
	starpu_codelet_init(cl);
	return cl;
}

void fstarpu_codelet_free(struct starpu_codelet *cl)
{
	memset(cl, 0, sizeof(*cl));
	free(cl);
}

void fstarpu_codelet_set_name(struct starpu_codelet *cl, const char *cl_name)
{
	cl->name = cl_name;
}

void fstarpu_codelet_add_cpu_func(struct starpu_codelet *cl, void *f_ptr)
{
	const size_t max_cpu_funcs = sizeof(cl->cpu_funcs)/sizeof(cl->cpu_funcs[0])-1;
	size_t i;
	for (i = 0; i < max_cpu_funcs; i++)
	{
		if (cl->cpu_funcs[i] == NULL)
		{
			cl->cpu_funcs[i] = f_ptr;
			return;
		}
	}
	_STARPU_ERROR("fstarpu: too many cpu functions in Fortran codelet");
}

void fstarpu_codelet_add_cuda_func(struct starpu_codelet *cl, void *f_ptr)
{
	const size_t max_cuda_funcs = sizeof(cl->cuda_funcs)/sizeof(cl->cuda_funcs[0])-1;
	unsigned i;
	for (i = 0; i < max_cuda_funcs; i++)
	{
		if (cl->cuda_funcs[i] == NULL)
		{
			cl->cuda_funcs[i] = f_ptr;
			return;
		}
	}
	_STARPU_ERROR("fstarpu: too many cuda functions in Fortran codelet");
}

void fstarpu_codelet_add_cuda_flags(struct starpu_codelet *cl, intptr_t flags)
{
	const size_t max_cuda_flags = sizeof(cl->cuda_flags)/sizeof(cl->cuda_flags[0])-1;
	unsigned i;
	for (i = 0; i < max_cuda_flags; i++)
	{
		if (cl->cuda_flags[i] == 0)
		{
			cl->cuda_flags[i] = (char)flags;
			return;
		}
	}
	_STARPU_ERROR("fstarpu: too many cuda flags in Fortran codelet");
}

void fstarpu_codelet_add_opencl_func(struct starpu_codelet *cl, void *f_ptr)
{
	const size_t max_opencl_funcs = sizeof(cl->opencl_funcs)/sizeof(cl->opencl_funcs[0])-1;
	unsigned i;
	for (i = 0; i < max_opencl_funcs; i++)
	{
		if (cl->opencl_funcs[i] == NULL)
		{
			cl->opencl_funcs[i] = f_ptr;
			return;
		}
	}
	_STARPU_ERROR("fstarpu: too many opencl functions in Fortran codelet");
}

void fstarpu_codelet_add_opencl_flags(struct starpu_codelet *cl, intptr_t flags)
{
	const size_t max_opencl_flags = sizeof(cl->opencl_flags)/sizeof(cl->opencl_flags[0])-1;
	unsigned i;
	for (i = 0; i < max_opencl_flags; i++)
	{
		if (cl->opencl_flags[i] == 0)
		{
			cl->opencl_flags[i] = (char)flags;
			return;
		}
	}
	_STARPU_ERROR("fstarpu: too many opencl flags in Fortran codelet");
}

void fstarpu_codelet_add_mic_func(struct starpu_codelet *cl, void *f_ptr)
{
	const size_t max_mic_funcs = sizeof(cl->mic_funcs)/sizeof(cl->mic_funcs[0])-1;
	unsigned i;
	for (i = 0; i < max_mic_funcs; i++)
	{
		if (cl->mic_funcs[i] == NULL)
		{
			cl->mic_funcs[i] = f_ptr;
			return;
		}
	}
	_STARPU_ERROR("fstarpu: too many mic functions in Fortran codelet");
}

void fstarpu_codelet_add_buffer(struct starpu_codelet *cl, intptr_t _mode)
{

	enum starpu_data_access_mode mode = (enum starpu_data_access_mode) _mode;
	const size_t max_modes = sizeof(cl->modes)/sizeof(cl->modes[0])-1;
	if ((mode & (STARPU_ACCESS_MODE_MAX-1)) != mode)
	{
		_STARPU_ERROR("fstarpu: invalid data mode");
	}
	if  (cl->nbuffers < (int) max_modes)
	{
		cl->modes[cl->nbuffers] = (unsigned int)mode;
		cl->nbuffers++;
	}
	else
	{
		_STARPU_ERROR("fstarpu: too many buffers in Fortran codelet");
	}
}

void fstarpu_codelet_set_variable_nbuffers(struct starpu_codelet *cl)
{
	cl->nbuffers = STARPU_VARIABLE_NBUFFERS;
}

void fstarpu_codelet_set_nbuffers(struct starpu_codelet *cl, int nbuffers)
{
	if (nbuffers >= 0)
	{
		cl->nbuffers = nbuffers;
	}
	else
	{
		_STARPU_ERROR("fstarpu: invalid nbuffers parameter");
	}
}

void fstarpu_codelet_set_flags(struct starpu_codelet *cl, intptr_t flags)
{
	cl->flags = (int)flags;
}

void fstarpu_codelet_set_where(struct starpu_codelet *cl, intptr_t where)
{
	STARPU_ASSERT(where >= 0);
	cl->where = (uint32_t)where;
}

void * fstarpu_variable_get_ptr(void *buffers[], int i)
{
	return (void *)STARPU_VARIABLE_GET_PTR(buffers[i]);
}

void * fstarpu_vector_get_ptr(void *buffers[], int i)
{
	return (void *)STARPU_VECTOR_GET_PTR(buffers[i]);
}

int fstarpu_vector_get_nx(void *buffers[], int i)
{
	return STARPU_VECTOR_GET_NX(buffers[i]);
}

void * fstarpu_matrix_get_ptr(void *buffers[], int i)
{
	return (void *)STARPU_MATRIX_GET_PTR(buffers[i]);
}

int fstarpu_matrix_get_ld(void *buffers[], int i)
{
	return STARPU_MATRIX_GET_LD(buffers[i]);
}

int fstarpu_matrix_get_nx(void *buffers[], int i)
{
	return STARPU_MATRIX_GET_NX(buffers[i]);
}

int fstarpu_matrix_get_ny(void *buffers[], int i)
{
	return STARPU_MATRIX_GET_NY(buffers[i]);
}

void * fstarpu_block_get_ptr(void *buffers[], int i)
{
	return (void *)STARPU_BLOCK_GET_PTR(buffers[i]);
}

int fstarpu_block_get_ldy(void *buffers[], int i)
{
	return STARPU_BLOCK_GET_LDY(buffers[i]);
}

int fstarpu_block_get_ldz(void *buffers[], int i)
{
	return STARPU_BLOCK_GET_LDZ(buffers[i]);
}

int fstarpu_block_get_nx(void *buffers[], int i)
{
	return STARPU_BLOCK_GET_NX(buffers[i]);
}

int fstarpu_block_get_ny(void *buffers[], int i)
{
	return STARPU_BLOCK_GET_NY(buffers[i]);
}

int fstarpu_block_get_nz(void *buffers[], int i)
{
	return STARPU_BLOCK_GET_NZ(buffers[i]);
}

void fstarpu_data_acquire(starpu_data_handle_t handle, intptr_t mode)
{
	STARPU_ASSERT(mode == fstarpu_r || mode == fstarpu_w || mode == fstarpu_rw);
	starpu_data_acquire(handle, (int)mode);
}

void fstarpu_unpack_arg(char *cl_arg, void **buffer_list)
{
	size_t current_arg_offset = 0;
	int nargs, arg;

	/* We fill the different pointers with the appropriate arguments */
	memcpy(&nargs, cl_arg, sizeof(nargs));
	current_arg_offset += sizeof(nargs);

	for (arg = 0; arg < nargs; arg++)
	{
		void *argptr = buffer_list[arg];

		/* If not reading all cl_args */
		if(argptr == NULL)
			break;

		size_t arg_size;
		memcpy(&arg_size, cl_arg+current_arg_offset, sizeof(arg_size));
		current_arg_offset += sizeof(arg_size);

		memcpy(argptr, cl_arg+current_arg_offset, arg_size);
		current_arg_offset += arg_size;
	}
}

void fstarpu_sched_ctx_display_workers(int ctx)
{
	starpu_sched_ctx_display_workers((unsigned)ctx, stderr);
}

intptr_t fstarpu_worker_get_type(int workerid)
{
	return (intptr_t)starpu_worker_get_type(workerid);
}

int fstarpu_worker_get_count_by_type(intptr_t type)
{
	return starpu_worker_get_count_by_type((enum starpu_worker_archtype)type);
}

unsigned fstarpu_worker_get_ids_by_type(intptr_t type, int *workerids, unsigned maxsize)
{
	return starpu_worker_get_ids_by_type((enum starpu_worker_archtype)type, workerids, maxsize);
}

int fstarpu_worker_get_by_type(intptr_t type, int num)
{
	return starpu_worker_get_by_type((enum starpu_worker_archtype)type, num);
}

int fstarpu_worker_get_by_devid(intptr_t type, int devid)
{
	return starpu_worker_get_by_type((enum starpu_worker_archtype)type, devid);
}

void fstarpu_worker_get_type_as_string(intptr_t type, char *dst, size_t maxlen)
{
	const char *str = starpu_worker_get_type_as_string((enum starpu_worker_archtype)type);
	snprintf(dst, maxlen, "%s", str);
}

starpu_data_handle_t *fstarpu_data_handle_array_alloc(int nb)
{
	void *ptr;
	_STARPU_CALLOC(ptr, (size_t)nb, sizeof(starpu_data_handle_t));
	return ptr;
}

void fstarpu_data_handle_array_free(starpu_data_handle_t *handles)
{
	free(handles);
}

void fstarpu_data_handle_array_set(starpu_data_handle_t *handles, int i, starpu_data_handle_t handle)
{
	handles[i] = handle;
}

struct starpu_data_descr *fstarpu_data_descr_array_alloc(int nb)
{
	void *ptr;
	_STARPU_CALLOC(ptr, (size_t)nb, sizeof(struct starpu_data_descr));
	return ptr;
}

struct starpu_data_descr *fstarpu_data_descr_alloc(void)
{
	return fstarpu_data_descr_array_alloc(1);
}

void fstarpu_data_descr_array_free(struct starpu_data_descr *descrs)
{
	free(descrs);
}

void fstarpu_data_descr_free(struct starpu_data_descr *descr)
{
	fstarpu_data_descr_array_free(descr);
}

void fstarpu_data_descr_array_set(struct starpu_data_descr *descrs, int i, starpu_data_handle_t handle, intptr_t mode)
{
	descrs[i].handle = handle;
	descrs[i].mode = (enum starpu_data_access_mode)mode;
}

void fstarpu_data_descr_set(struct starpu_data_descr *descr, starpu_data_handle_t handle, intptr_t mode)
{
	fstarpu_data_descr_array_set(descr, 1, handle, mode);
}

struct starpu_data_filter *fstarpu_data_filter_allocate(void)
{
	struct starpu_data_filter *filter;
	_STARPU_CALLOC(filter, 1, sizeof(*filter));
	return filter;
}

/* Note: use fstarpu_df_alloc_ prefix instead of fstarpu_data_filter_allocate_ to fit within the
 * Fortran id length limit */
#define _FSTARPU_DATA_FILTER_ALLOCATOR(name) \
struct starpu_data_filter *fstarpu_df_alloc_##name(void) \
{ \
	struct starpu_data_filter *filter = fstarpu_data_filter_allocate(); \
	filter->filter_func = starpu_##name; \
	return filter; \
}

_FSTARPU_DATA_FILTER_ALLOCATOR(bcsr_filter_canonical_block);
_FSTARPU_DATA_FILTER_ALLOCATOR(csr_filter_vertical_block);
_FSTARPU_DATA_FILTER_ALLOCATOR(matrix_filter_block);
_FSTARPU_DATA_FILTER_ALLOCATOR(matrix_filter_block_shadow);
_FSTARPU_DATA_FILTER_ALLOCATOR(matrix_filter_vertical_block);
_FSTARPU_DATA_FILTER_ALLOCATOR(matrix_filter_vertical_block_shadow);
_FSTARPU_DATA_FILTER_ALLOCATOR(vector_filter_block);
_FSTARPU_DATA_FILTER_ALLOCATOR(vector_filter_block_shadow);
_FSTARPU_DATA_FILTER_ALLOCATOR(vector_filter_list);
_FSTARPU_DATA_FILTER_ALLOCATOR(vector_filter_divide_in_2);
_FSTARPU_DATA_FILTER_ALLOCATOR(block_filter_block);
_FSTARPU_DATA_FILTER_ALLOCATOR(block_filter_block_shadow);
_FSTARPU_DATA_FILTER_ALLOCATOR(block_filter_vertical_block);
_FSTARPU_DATA_FILTER_ALLOCATOR(block_filter_vertical_block_shadow);

#undef _FSTARPU_DATA_FILTER_ALLOCATOR

void fstarpu_data_filter_free(struct starpu_data_filter *filter)
{
	memset(filter, 0, sizeof(*filter));
	free(filter);
}

void fstarpu_data_filter_set_filter_func(struct starpu_data_filter *filter, void *f_ptr)
{
	STARPU_ASSERT(f_ptr != NULL);
	filter->filter_func = f_ptr;
}

void fstarpu_data_filter_set_nchildren(struct starpu_data_filter *filter, int nchildren)
{
	STARPU_ASSERT(nchildren >= 0);
	filter->nchildren = nchildren;
}

void fstarpu_data_filter_set_get_nchildren_func(struct starpu_data_filter *filter, void *f_ptr)
{
	filter->get_nchildren = f_ptr;
}

void fstarpu_data_filter_set_get_child_ops_func(struct starpu_data_filter *filter, void *f_ptr)
{
	filter->get_child_ops = f_ptr;
}

void fstarpu_data_filter_set_filter_arg(struct starpu_data_filter *filter, int filter_arg)
{
	STARPU_ASSERT(filter_arg >= 0); /* starpu_data_filter.filter_arg is unsigned, but
					 * Fortran does not support unsigned types */
	filter->filter_arg = (unsigned)filter_arg;
}

void fstarpu_data_filter_set_filter_arg_ptr(struct starpu_data_filter *filter, void *filter_arg_ptr)
{
	filter->filter_arg_ptr = filter_arg_ptr;
}
