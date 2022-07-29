/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2022 Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/config.h>
#include <omp.h>
#ifdef STARPU_OPENMP_LLVM
#include <stdlib.h>
#include <stdio.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <assert.h>
#include <util/openmp_runtime_support.h>

typedef struct ident ident_t;
typedef int32_t kmp_int32;
typedef void * kmp_intptr_t;

typedef void(* kmpc_micro) (kmp_int32 *global_tid, kmp_int32 *bound_tid,...);

typedef kmp_int32 (*kmp_routine_entry_t)(kmp_int32 gtid, void *kmp_task);

typedef struct kmp_depend_info
{
	kmp_intptr_t base_addr;
	size_t len;
	struct
	{
		bool in : 1;
		bool out : 1;
	} flags;
	size_t elem_size;
} kmp_depend_info_t;

typedef union kmp_cmplrdata
{
	kmp_int32 priority; /**< priority specified by user for the task */
	kmp_routine_entry_t destructors; /* pointer to function to invoke deconstructors of firstprivate C++ objects */
	/* future data */
} kmp_cmplrdata_t;

typedef void *(*kmp_variant_entry_t)(void *, ...);

typedef enum kmp_variant_kind
{
 	VARIANT_CPU,
	VARIANT_OPENCL,
	VARIANT_CUDA
} kmp_variant_kind_t;

typedef struct kmp_variant
{
	kmp_variant_entry_t fn;
	kmp_variant_kind_t kind;
} kmp_variant_t;

typedef struct kmp_task
{ /* GEH: Shouldn't this be aligned somehow? */
	void *shareds; /**< pointer to block of pointers to shared vars   */
	kmp_routine_entry_t routine; /**< pointer to routine to call for executing task */
	kmp_int32 part_id; /**< part id for the task                          */
	kmp_cmplrdata_t data1; /* Two known optional additions: destructors and priority */
	kmp_cmplrdata_t data2; /* Process destructors first, priority second */
	/* future data */
	kmp_variant_t *variants;
	kmp_int32 nvariants;
} kmp_task_t;

struct s_microtask_wrap
{
	int argc;
	void **arg_ptrs;
	kmpc_micro microtask;
};

enum sched_type /* : kmp_int32 */
{
	kmp_sch_lower	= 32,
	kmp_sch_static_chunked	= 33,
	kmp_sch_static	= 34,
	kmp_sch_dynamic_chunked	= 35,
	kmp_sch_guided_chunked	= 36,
	kmp_sch_runtime	= 37,
	kmp_sch_auto	= 38,
	kmp_sch_trapezoidal	= 39,

	kmp_sch_static_greedy	= 40,
	kmp_sch_static_balanced	= 41,

	kmp_sch_guided_iterative_chunked	= 42,
	kmp_sch_guided_analytical_chunked	= 43,

	kmp_sch_static_steal	= 44,

	kmp_sch_static_balanced_chunked	= 45,
	kmp_sch_guided_simd	= 46,
	kmp_sch_runtime_simd	= 47,

	kmp_sch_upper,

	kmp_ord_lower	= 64,
	kmp_ord_static_chunked	= 65,
	kmp_ord_static	= 66,
	kmp_ord_dynamic_chunked	= 67,
	kmp_ord_guided_chunked	= 68,
	kmp_ord_runtime	= 69,
	kmp_ord_auto	= 70,
	kmp_ord_trapezoidal	= 71,
	kmp_ord_upper,

	kmp_distribute_static_chunked	= 91,
	kmp_distribute_static	= 92,

	kmp_nm_lower	= 160,

	kmp_nm_static_chunked	= (kmp_sch_static_chunked - kmp_sch_lower + kmp_nm_lower),
	kmp_nm_static	= 162,
	kmp_nm_dynamic_chunked	= 163,
	kmp_nm_guided_chunked	= 164,
	kmp_nm_runtime	= 165,
	kmp_nm_auto	= 166,
	kmp_nm_trapezoidal	= 167,

	kmp_nm_static_greedy	= 168,
	kmp_nm_static_balanced	= 169,
	kmp_nm_guided_iterative_chunked	= 170,
	kmp_nm_guided_analytical_chunked	= 171,
	kmp_nm_static_steal	= 172,

	kmp_nm_ord_static_chunked	= 193,
	kmp_nm_ord_static	= 194,
	kmp_nm_ord_dynamic_chunked	= 195,
	kmp_nm_ord_guided_chunked	= 196,
	kmp_nm_ord_runtime	= 197,
	kmp_nm_ord_auto	= 198,
	kmp_nm_ord_trapezoidal	= 199,
	kmp_nm_upper,

	kmp_sch_modifier_monotonic	= (1 << 29),
	kmp_sch_modifier_nonmonotonic	= (1 << 30),

	kmp_sch_default	= kmp_sch_static
};

typedef kmp_int32 kmp_critical_name[8];

kmp_int32 __kmpc_global_thread_num(ident_t *loc);
kmp_int32 __kmpc_global_num_threads(ident_t *loc);
kmp_int32 __kmpc_bound_thread_num(ident_t *loc);
kmp_int32 __kmpc_bound_num_threads(ident_t *loc);

static void parallel_call(void *buffers[], void *args)
{
	(void) buffers;
	int gtid=__kmpc_global_thread_num(NULL);
	int ltid=__kmpc_bound_thread_num(NULL);
	void **arg_ptrs = args;
	kmpc_micro microtask = *arg_ptrs++;
	kmp_int32 argc = (intptr_t)*arg_ptrs++;
	switch (argc)
	{
	case 0:
		microtask(&gtid, &ltid);
		break;

	case 1:
		microtask(&gtid, &ltid, arg_ptrs[0]);
		break;

	case 2:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1]);
		break;

	case 3:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2]);
		break;

	case 4:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3]);
		break;

	case 5:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4]);
		break;

	case 6:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5]);
		break;

	case 7:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6]);
		break;

	case 8:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7]);
		break;

	case 9:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8]);
		break;

	case 10:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8], arg_ptrs[9]);
		break;

	case 11:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8], arg_ptrs[9], arg_ptrs[10]);
		break;

	case 12:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8], arg_ptrs[9], arg_ptrs[10], arg_ptrs[11]);
		break;

	case 13:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8], arg_ptrs[9], arg_ptrs[10], arg_ptrs[11], arg_ptrs[12]);
		break;

	case 14:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8], arg_ptrs[9], arg_ptrs[10], arg_ptrs[11], arg_ptrs[12], arg_ptrs[13]);
		break;

	case 15:
		microtask(&gtid, &ltid, arg_ptrs[0], arg_ptrs[1], arg_ptrs[2], arg_ptrs[3], arg_ptrs[4], arg_ptrs[5], arg_ptrs[6], arg_ptrs[7], arg_ptrs[8], arg_ptrs[9], arg_ptrs[10], arg_ptrs[11], arg_ptrs[12], arg_ptrs[13], arg_ptrs[14]);
		break;

	default:
		assert(0);
	}
}

/* Deprecated Functions */
kmp_int32 __kmpc_ok_to_fork(ident_t *loc)
{
	(void) loc;
	return !0;
}

/* Startup and Shutdown */
void __kmpc_begin(ident_t *loc, kmp_int32 flags)
{
	(void) loc;
	(void) flags;
	/* TODO: add auto-init in other lib funcs if kmpc_begin is not called */
	starpu_omp_init();
}

void __kmpc_end(ident_t *loc)
{
	(void) loc;
	/* TODO: add support for KMP_IGNORE_MPPEND */
	starpu_omp_shutdown();
}

/* Parallel (fork/join) */
void __kmpc_push_num_threads(ident_t *loc, kmp_int32 global_tid, kmp_int32 num_threads)
{
	(void) loc;
	(void) global_tid;
	(void) num_threads;
	abort();
}

void __kmpc_fork_call(ident_t *loc, kmp_int32 argc, kmpc_micro microtask, ...)
{
	(void) loc;
	va_list vargs;
	va_start(vargs, microtask);
	void *arg_ptrs[2+argc];
	arg_ptrs[0] = microtask;
	arg_ptrs[1] = (void*)(intptr_t)argc;

	int i;
	for (i=0; i<argc; i++)
	{
		arg_ptrs[i+2] = va_arg(vargs, void*);
	}

	struct starpu_omp_parallel_region_attr *attr = calloc(1, sizeof(struct starpu_omp_parallel_region_attr));
#ifdef STARPU_SIMGRID
	attr->cl.model        = &starpu_perfmodel_nop;
	attr->cl.flags        = STARPU_CODELET_SIMGRID_EXECUTE;
#endif
	attr->cl.cpu_funcs[0] = parallel_call;
	attr->cl.where        = STARPU_CPU;
	attr->cl_arg_size     = (argc+2)*sizeof(void *);
	attr->cl_arg_free     = 0;
	attr->cl_arg          = arg_ptrs;
	attr->if_clause       = 1;
	starpu_omp_parallel_region(attr);
	free((void *)attr);

	va_end(vargs);
}

static void task_call(void *buffers[], void *args)
{
	(void) buffers;
	int gtid=__kmpc_global_thread_num(NULL);
	void **arg_ptrs = args;
	kmp_task_t *task = *arg_ptrs++;
	/*typedef kmp_int32 (*kmp_routine_entry_t)(kmp_int32 gtid, void *kmp_task);*/
	task->routine(gtid, task);
}

kmp_task_t *__kmpc_omp_task_alloc(ident_t *loc_ref, kmp_int32 gtid,
				  kmp_int32 flags, size_t sizeof_kmp_task_t,
				  size_t sizeof_shareds,
				  kmp_routine_entry_t task_entry)
{
	(void) loc_ref;
	(void) gtid;
	(void) flags;
	// The initial content of kmp_task_t is:
	// - void *shared
	// - kmp_routine_entry_t routine
	// - kmp_int32 part_id
	// But the compiler may need more fields, hence it passes a "sizeof_kmp_task_t" that we should honor.
	kmp_task_t *task;
	/* FIXME: avoid double malloc by allocating shared+task_t at once */
	/* FIXME: free the things somewhere*/
	_STARPU_MALLOC(task, sizeof_kmp_task_t);
	void *shared;
	_STARPU_MALLOC(shared, sizeof_shareds);
	task->shareds = shared;
	task->routine = task_entry;
	task->part_id = 0;
	task->variants = 0;
	task->nvariants = 0;
	return task;
}

#define GETDEP(task, i) starpu_data_handle_to_pointer(task->starpu_task->handles[i], STARPU_MAIN_RAM)
#define GET(i) (void*)STARPU_VARIABLE_GET_PTR(buffers[i])

static void task_call_variants(void (*fn)(void*, ...), void *buffers[], void *args)
{
	void **arg_ptrs = args;
	int nargs = arg_ptrs[1];
	// TODO: asm it, as we could do it nicely in a loop
	switch (nargs)
	{
	case 0:
		fn(0);
		break;
	case 1:
		fn(GET(0));
		break;
	case 2:
		fn(GET(0), GET(1));
		break;
	case 3:
		fn(GET(0), GET(1), GET(2));
		break;
	case 4:
		fn(GET(0), GET(1), GET(2), GET(3));
		break;
	case 5:
		fn(GET(0), GET(1), GET(2), GET(3), GET(4));
		break;
	case 6:
		fn(GET(0), GET(1), GET(2), GET(3), GET(4), GET(5));
		break;
	default:
		fprintf(stderr, "Unsupported number of dependencies/arguments in task call.\n");
		abort();
		break;
	}
}
#undef GETDEP

static void task_call_cpu(void *buffers[], void *args)
{
	void **arg_ptrs = args;
	task_call_variants((void (*)(void *, ...))arg_ptrs[2], buffers, args);
}

static void task_call_cuda(void *buffers[], void *args)
{
	void **arg_ptrs = args;
	task_call_variants((void (*)(void *, ...))arg_ptrs[3], buffers, args);
}

/*TODO: wrapper void *(buffers[], nbuffer) { push push call }*/

kmp_task_t *__kmpc_omp_task_alloc_variants(ident_t *loc_ref, kmp_int32 gtid,
                                           kmp_int32 flags,
                                           size_t sizeof_kmp_task_t,
                                           size_t sizeof_shareds,
                                           kmp_routine_entry_t task_entry,
                                           kmp_int32 nvariants)
{
	kmp_task_t *task = __kmpc_omp_task_alloc(loc_ref, gtid, flags, sizeof_kmp_task_t, sizeof_shareds, task_entry);
	task->nvariants = nvariants;
	_STARPU_MALLOC(task->variants, nvariants * sizeof(kmp_variant_t));
	return task;
}

kmp_int32 __kmpc_omp_taskwait(ident_t *loc_ref, kmp_int32 gtid)
{
	(void) loc_ref;
	(void) gtid;
	starpu_omp_taskwait();
	return 0;
}

kmp_int32 __kmpc_omp_task_with_deps(ident_t *loc_ref, kmp_int32 gtid,
				    kmp_task_t * new_task, kmp_int32 ndeps,
				    kmp_depend_info_t *dep_list,
				    kmp_int32 ndeps_noalias,
				    kmp_depend_info_t *noalias_dep_list)
{
	(void) loc_ref;
	(void) gtid;

	/* NOTE: for some reason, just having a static struct and passing its address
	 * triggered a segfault in the starpu_omp_task_region.
	 * */
	static int _msg=0;
	if (_msg == 0)
	{
		_STARPU_MSG("Using the StarPU OpenMP LLVM Support\n");
		_msg = 1;
	}

	struct starpu_omp_task_region_attr *attr = calloc(1, sizeof(struct starpu_omp_task_region_attr));

	/* This is freed in starpu_omp_task_region, as attr.cl_arg_free is set to true*/
	void **arg_ptrs = calloc(4, sizeof(void*));
	arg_ptrs[0] = new_task;
	arg_ptrs[1] = ndeps + ndeps_noalias;

	if (new_task->nvariants == 0)
	{
		attr->cl.cpu_funcs[0]  = task_call;
		attr->cl.where         = STARPU_CPU;
	}
	else
	{
		for (int i = 0; i < new_task->nvariants; ++i)
		{
			switch(new_task->variants[i].kind)
			{
			case VARIANT_CPU:
				attr->cl.where         |= STARPU_CPU;
				attr->cl.cpu_funcs[0] = task_call_cpu;
				arg_ptrs[2] = new_task->variants[i].fn;
				break;
			case VARIANT_CUDA:
				attr->cl.where         |= STARPU_CUDA;
				attr->cl.cuda_funcs[0] = task_call_cuda;
				arg_ptrs[3] = new_task->variants[i].fn;
				break;
			case VARIANT_OPENCL:
				fprintf(stderr, "variant for opencl detected but not supported: %p, ignoring.\n", new_task->variants[i].fn);
				break;
			}
		}
	}

	attr->cl_arg_size     = (4)*sizeof(void *);
	attr->cl_arg_free     = 1;
	attr->cl_arg          = arg_ptrs;
	attr->if_clause        = 1;
	attr->final_clause     = 0;
	attr->untied_clause    = 1;
	attr->mergeable_clause = 0;
	attr->cl.nbuffers = ndeps + ndeps_noalias;
	starpu_data_handle_t *handles = calloc(attr->cl.nbuffers, sizeof(starpu_data_handle_t));
	int current_buffer = 0;
	starpu_data_handle_t current_handler = 0;
	for (int i = 0; i < ndeps; i++)
	{
		if (dep_list[i].flags.in && dep_list[i].flags.out)
		{
			attr->cl.modes[current_buffer] = STARPU_RW;
		}
		else if (dep_list[i].flags.in)
		{
			attr->cl.modes[current_buffer] = STARPU_R;
		}
		else
		{
			attr->cl.modes[current_buffer] = STARPU_W;
		}
		current_handler = starpu_omp_data_lookup(dep_list[i].base_addr);
		if (current_handler)
		{
			handles[current_buffer] = current_handler;
		}
		else
		{
			if (dep_list[i].len == 1)
			{
				starpu_variable_data_register(&handles[current_buffer], STARPU_MAIN_RAM, (uintptr_t)dep_list[i].base_addr, sizeof(kmp_intptr_t));
				starpu_omp_handle_register(handles[current_buffer], dep_list[i].base_addr);
			}
			else
			{
				starpu_vector_data_register(&handles[current_buffer], STARPU_MAIN_RAM, (uintptr_t)dep_list[i].base_addr, dep_list[i].len, dep_list[i].elem_size);
				starpu_omp_handle_register(handles[current_buffer], dep_list[i].base_addr);
			}
		}
		current_buffer++;
	}
	for (int i = 0; i < ndeps_noalias; i++)
	{
		if (noalias_dep_list[i].flags.in && noalias_dep_list[i].flags.out)
		{
			attr->cl.modes[current_buffer] = STARPU_RW;
		}
		else if (noalias_dep_list[i].flags.in)
		{
			attr->cl.modes[current_buffer] = STARPU_R;
		}
		else
		{
			attr->cl.modes[current_buffer] = STARPU_W;
		}
		current_handler = starpu_omp_data_lookup(noalias_dep_list[i].base_addr);
		if (current_handler)
		{
			handles[current_buffer] = current_handler;
		}
		else
		{
			if (dep_list[i].len == 1)
			{
				starpu_variable_data_register(&handles[current_buffer], STARPU_MAIN_RAM, (uintptr_t)dep_list[i].base_addr, sizeof(kmp_intptr_t));
				starpu_omp_handle_register(handles[current_buffer], dep_list[i].base_addr);
			}
			else
			{
				starpu_vector_data_register(&handles[current_buffer], STARPU_MAIN_RAM, (uintptr_t)dep_list[i].base_addr, dep_list[i].len, dep_list[i].elem_size);
				starpu_omp_handle_register(handles[current_buffer], dep_list[i].base_addr);
			}
		}
		current_buffer++;
	}

	if (current_buffer)
	{
		// If we have any deps
		attr->handles = &handles[0];
	}

	// thoughts : create starpu_omp_task_region_attr here, fill it with kmp_taskdata
	// keep an arg to the wrapper with the kmp_task_t
	starpu_omp_task_region(attr);
	free(attr);
	return 0;
}

kmp_int32 __kmpc_omp_task(ident_t *loc_ref, kmp_int32 gtid, kmp_task_t *new_task)
{
	int retval = __kmpc_omp_task_with_deps(loc_ref, gtid, new_task, 0, 0, 0, 0);
	return retval;
}

void __kmpc_push_num_teams(ident_t *loc, kmp_int32 global_tid, kmp_int32 num_teams, kmp_int32 num_threads)
{
	(void) loc;
	(void) global_tid;
	(void) num_teams;
	(void) num_threads;
	abort();
}

void __kmpc_fork_teams(ident_t *loc, kmp_int32 argc, kmpc_micro microtask, ...)
{
	(void) loc;
	(void) argc;
	(void) microtask;
	abort();
}

void __kmpc_serialized_parallel(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	abort();
}

void __kmpc_end_serialized_parallel(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	abort();
}

/* Thread Information */
kmp_int32 __kmpc_global_thread_num(ident_t *loc)
{
	(void) loc;
	struct starpu_omp_region *region;
	region = _starpu_omp_get_region_at_level(1);
	if (region == NULL)
		return 0;
	return _starpu_omp_get_region_thread_num(region);
}

kmp_int32 __kmpc_global_num_threads(ident_t *loc)
{
	(void) loc;
	struct starpu_omp_region *region;
	region = _starpu_omp_get_region_at_level(1);
	if (region == NULL)
		return 1;
	return region->nb_threads;
}

kmp_int32 __kmpc_bound_thread_num(ident_t *loc)
{
	(void) loc;
	return starpu_omp_get_thread_num();
}

kmp_int32 __kmpc_bound_num_threads(ident_t *loc)
{
	(void) loc;
	return starpu_omp_get_num_threads();
}

kmp_int32 __kmpc_in_parallel(ident_t *loc)
{
	(void) loc;
	return starpu_omp_in_parallel();
}

/* Work sharing */
kmp_int32 __kmpc_master(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	return starpu_omp_master_inline();
}

void __kmpc_end_master(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	/* nothing */
}

void __kmpc_ordered(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	starpu_omp_ordered_inline_begin();
}

void __kmpc_end_ordered(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	starpu_omp_ordered_inline_end();
}

kmp_int32 __kmpc_single(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	return starpu_omp_single_inline();
}

void __kmpc_end_single(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	/* nothing */
}

void __kmpc_dispatch_init_4(ident_t *loc)
{
	(void) loc;
	abort();
}

void __kmpc_dispatch_next_4(ident_t *loc)
{
	(void) loc;
	abort();
}

/* Work sharing */
void __kmpc_flush(ident_t *loc)
{
	(void) loc;
	abort();
}

void __kmpc_barrier(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	starpu_omp_barrier();
}

kmp_int32 __kmpc_barrier_master(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	abort();
}

void __kmpc_end_barrier_master(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	abort();
}

kmp_int32 __kmpc_barrier_master_nowait(ident_t *loc, kmp_int32 global_tid)
{
	(void) loc;
	(void) global_tid;
	abort();
}

void __kmpc_reduce_nowait(ident_t *loc, kmp_int32 global_tid, kmp_int32 num_vars, size_t reduce_size, void *reduce_data, void (*reduce_func)(void *lhs_data, void *rhs_data), kmp_critical_name *lck)
{
	(void) loc;
	(void) global_tid;
	(void) num_vars;
	(void) reduce_size;
	(void) reduce_data;
	(void) reduce_func;
	(void) lck;
	abort();
}

void __kmpc_end_reduce_nowait(ident_t *loc, kmp_int32 global_tid, kmp_critical_name *lck)
{
	(void) loc;
	(void) global_tid;
	(void) lck;
	abort();
}

void __kmpc_reduce(ident_t *loc, kmp_int32 global_tid, kmp_int32 num_vars, size_t reduce_size, void *reduce_data, void (*reduce_func)(void *lhs_data, void *rhs_data), kmp_critical_name *lck)
{
	(void) loc;
	(void) global_tid;
	(void) num_vars;
	(void) reduce_size;
	(void) reduce_data;
	(void) reduce_func;
	(void) lck;
	abort();
}

void __kmpc_end_reduce(ident_t *loc, kmp_int32 global_tid, kmp_critical_name *lck)
{
	(void) loc;
	(void) global_tid;
	(void) lck;
	abort();
}

/* lib constructor/destructor */

__attribute__((constructor))
static void __kmp_constructor(void)
{
	static int _msg=0;
	if (_msg == 0)
	{
		_STARPU_MSG("Initialising the StarPU OpenMP LLVM Support\n");
		_msg = 1;
	}

	int ret = starpu_omp_init();
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_omp_init");
}

__attribute__((destructor))
static void kmp_destructor(void)
{
	starpu_omp_shutdown();
}

/* omp lib API */

void omp_set_num_threads(int threads)
{
	starpu_omp_set_num_threads(threads);
}

int omp_get_num_threads()
{
	return starpu_omp_get_num_threads();
}

int omp_get_thread_num()
{
	return starpu_omp_get_thread_num();
}

int omp_get_max_threads()
{
	return starpu_omp_get_max_threads();
}

int omp_get_num_procs(void)
{
	return starpu_omp_get_num_procs();
}

int omp_in_parallel(void)
{
	return starpu_omp_in_parallel();
}

void omp_set_dynamic(int dynamic_threads)
{
	starpu_omp_set_dynamic(dynamic_threads);
}

int omp_get_dynamic(void)
{
	return starpu_omp_get_dynamic();
}

void omp_set_nested(int nested)
{
	starpu_omp_set_nested(nested);
}

int omp_get_nested(void)
{
	return starpu_omp_get_nested();
}

int omp_get_cancellation(void)
{
	return starpu_omp_get_cancellation();
}

void omp_set_schedule(enum omp_sched_value kind, int modifier)
{
	starpu_omp_set_schedule(kind, modifier);
}

void omp_get_schedule(enum omp_sched_value *kind, int *modifier)
{
	starpu_omp_get_schedule((enum starpu_omp_sched_value*)kind, modifier);
}

int omp_get_thread_limit(void)
{
	return starpu_omp_get_thread_limit();
}

void omp_set_max_active_levels(int max_levels)
{
	starpu_omp_set_max_active_levels(max_levels);
}

int omp_get_max_active_levels(void)
{
	return starpu_omp_get_max_active_levels();
}

int omp_get_level(void)
{
	return starpu_omp_get_level();
}

int omp_get_ancestor_thread_num(int level)
{
	return starpu_omp_get_ancestor_thread_num(level);
}

int omp_get_team_size(int level)
{
	return starpu_omp_get_team_size(level);
}

int omp_get_active_level(void)
{
	return starpu_omp_get_active_level();
}

int omp_in_final(void)
{
	return starpu_omp_in_final();
}

enum omp_proc_bind_value omp_get_proc_bind(void)
{
	return starpu_omp_get_proc_bind();
}

int omp_get_num_places(void)
{
	return starpu_omp_get_num_places();
}

int omp_get_place_num_procs(int place_num)
{
	return starpu_omp_get_place_num_procs(place_num);
}

void omp_get_place_proc_ids(int place_num, int *ids)
{
	starpu_omp_get_place_proc_ids(place_num, ids);
}

int omp_get_place_num(void)
{
	return starpu_omp_get_place_num();
}

int omp_get_partition_num_places(void)
{
	return starpu_omp_get_partition_num_places();
}

void omp_get_partition_place_nums(int *place_nums)
{
	starpu_omp_get_partition_place_nums(place_nums);
}

void omp_set_default_device(int device_num)
{
	starpu_omp_set_default_device(device_num);
}

int omp_get_default_device(void)
{
	return starpu_omp_get_default_device();
}

int omp_get_num_devices(void)
{
	return starpu_omp_get_num_devices();
}

int omp_get_num_teams(void)
{
	return starpu_omp_get_num_teams();
}

int omp_get_team_num(void)
{
	return starpu_omp_get_team_num();
}

int omp_is_initial_device(void)
{
	return starpu_omp_is_initial_device();
}

int omp_get_initial_device(void)
{
	return starpu_omp_get_initial_device();
}

int omp_get_max_task_priority(void)
{
	return starpu_omp_get_max_task_priority();
}

void omp_init_lock(omp_lock_t *lock)
{
	starpu_omp_init_lock(lock);
}

void omp_destroy_lock(omp_lock_t *lock)
{
	starpu_omp_destroy_lock(lock);
}

void omp_set_lock(omp_lock_t *lock)
{
	starpu_omp_set_lock(lock);
}

void omp_unset_lock(omp_lock_t *lock)
{
	starpu_omp_unset_lock(lock);
}

int omp_test_lock(omp_lock_t *lock)
{
	return starpu_omp_test_lock(lock);
}

void omp_init_nest_lock(omp_nest_lock_t *lock)
{
	starpu_omp_init_nest_lock(lock);
}

void omp_destroy_nest_lock(omp_nest_lock_t *lock)
{
	starpu_omp_destroy_nest_lock(lock);
}

void omp_set_nest_lock(omp_nest_lock_t *lock)
{
	starpu_omp_set_nest_lock(lock);
}

void omp_unset_nest_lock(omp_nest_lock_t *lock)
{
	starpu_omp_unset_nest_lock(lock);
}

int omp_test_nest_lock(omp_nest_lock_t *lock)
{
	return starpu_omp_test_nest_lock(lock);
}

double omp_get_wtime(void)
{
	return starpu_omp_get_wtime();
}

double omp_get_wtick(void)
{
	return starpu_omp_get_wtick();
}

void *omp_get_local_cuda_stream(void)
{
#ifdef STARPU_USE_CUDA
	return starpu_cuda_get_local_stream();
#else
	return 0;
#endif
}

#endif /* STARPU_OPENMP_LLVM */
