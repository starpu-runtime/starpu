/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <util/starpu_task_insert_utils.h>
#include <common/config.h>
#include <common/utils.h>
#include <core/task.h>

void starpu_codelet_pack_arg_init(struct starpu_codelet_pack_arg_data *state)
{
	state->arg_buffer = NULL;
	state->arg_buffer_size = 0;
	state->current_offset = sizeof(int);
	state->nargs = 0;
}

void starpu_codelet_pack_arg(struct starpu_codelet_pack_arg_data *state, const void *ptr, size_t ptr_size)
{
	STARPU_ASSERT_MSG(state->current_offset >= sizeof(int), "struct starpu_codelet_pack_arg has to be initialized with starpu_codelet_pack_arg_init");
	if (state->current_offset + sizeof(ptr_size) + ptr_size > state->arg_buffer_size)
	{
		if (state->arg_buffer_size == 0)
			state->arg_buffer_size = 128 + sizeof(ptr_size) + ptr_size;
		else
			state->arg_buffer_size = 2 * state->arg_buffer_size + sizeof(ptr_size) + ptr_size;
		_STARPU_REALLOC(state->arg_buffer, state->arg_buffer_size);
	}
	memcpy(state->arg_buffer+state->current_offset, (void *)&ptr_size, sizeof(ptr_size));
	state->current_offset += sizeof(ptr_size);

	memcpy(state->arg_buffer+state->current_offset, ptr, ptr_size);
	state->current_offset += ptr_size;
	STARPU_ASSERT(state->current_offset <= state->arg_buffer_size);
	state->nargs++;
}

void starpu_codelet_pack_arg_fini(struct starpu_codelet_pack_arg_data *state, void **cl_arg, size_t *cl_arg_size)
{
	if (state->nargs)
	{
		memcpy(state->arg_buffer, &state->nargs, sizeof(state->nargs));
	}
	else
	{
		free(state->arg_buffer);
		state->arg_buffer = NULL;
	}

	*cl_arg = state->arg_buffer;
	*cl_arg_size = state->arg_buffer_size;
}

int _starpu_codelet_pack_args(void **arg_buffer, size_t *arg_buffer_size, va_list varg_list)
{
	int arg_type;

	struct starpu_codelet_pack_arg_data state;
	starpu_codelet_pack_arg_init(&state);

	while((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type & STARPU_R || arg_type & STARPU_W || arg_type & STARPU_SCRATCH || arg_type & STARPU_REDUX)
		{
			(void)va_arg(varg_list, starpu_data_handle_t);
		}
		else if (arg_type==STARPU_DATA_ARRAY)
		{
			(void)va_arg(varg_list, starpu_data_handle_t*);
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_DATA_MODE_ARRAY)
		{
			(void)va_arg(varg_list, struct starpu_data_descr*);
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_VALUE)
		{
			/* We have a constant value: this should be followed by a pointer to the cst value and the size of the constant */
			void *ptr = va_arg(varg_list, void *);
			size_t ptr_size = va_arg(varg_list, size_t);

			starpu_codelet_pack_arg(&state, ptr, ptr_size);
		}
		else if (arg_type==STARPU_CL_ARGS)
		{
			(void)va_arg(varg_list, void *);
			(void)va_arg(varg_list, size_t);
		}
		else if (arg_type==STARPU_CL_ARGS_NFREE)
		{
			(void)va_arg(varg_list, void *);
			(void)va_arg(varg_list, size_t);
		}
		else if (arg_type==STARPU_TASK_DEPS_ARRAY)
		{
			(void)va_arg(varg_list, unsigned);
			(void)va_arg(varg_list, struct starpu_task **);
		}
		else if (arg_type==STARPU_TASK_END_DEPS_ARRAY)
		{
			(void)va_arg(varg_list, unsigned);
			(void)va_arg(varg_list, struct starpu_task **);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			(void)va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			va_arg(varg_list, _starpu_callback_func_t);
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG_NFREE)
		{
			va_arg(varg_list, _starpu_callback_func_t);
			va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			(void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_CALLBACK_ARG_NFREE)
		{
			(void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
		{
			va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
		{
			(void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG_NFREE)
		{
			(void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP)
		{
			va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG)
		{
			(void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE)
		{
			(void)va_arg(varg_list, void *);
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_DATA)
		{
			(void)va_arg(varg_list, starpu_data_handle_t);
		}
		else if (arg_type==STARPU_EXECUTE_WHERE)
		{
			(void)va_arg(varg_list, unsigned long long);
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_WORKER_ORDER)
		{
			va_arg(varg_list, unsigned);
		}
		else if (arg_type==STARPU_SCHED_CTX)
		{
			(void)va_arg(varg_list, unsigned);
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_POSSIBLY_PARALLEL)
		{
			(void)va_arg(varg_list, unsigned);
		}
		else if (arg_type==STARPU_FLOPS)
		{
			(void)va_arg(varg_list, double);
		}
		else if (arg_type==STARPU_TAG || arg_type==STARPU_TAG_ONLY)
		{
			(void)va_arg(varg_list, starpu_tag_t);
		}
		else if (arg_type==STARPU_NAME)
		{
			(void)va_arg(varg_list, const char *);
		}
		else if (arg_type==STARPU_NODE_SELECTION_POLICY)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_TASK_COLOR)
                {
                        (void)va_arg(varg_list, int);
                }
		else if (arg_type==STARPU_TASK_SYNCHRONOUS)
                {
                        (void)va_arg(varg_list, int);
                }
		else if (arg_type==STARPU_HANDLES_SEQUENTIAL_CONSISTENCY)
                {
                        (void)va_arg(varg_list, unsigned char *);
                }
		else if (arg_type==STARPU_TASK_END_DEP)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_TASK_WORKERIDS)
		{
			(void)va_arg(varg_list, unsigned);
			(void)va_arg(varg_list, uint32_t*);
		}
		else if (arg_type==STARPU_SEQUENTIAL_CONSISTENCY)
		{
			(void)va_arg(varg_list, unsigned);
		}
		else if (arg_type==STARPU_TASK_PROFILING_INFO)
		{
			(void)va_arg(varg_list, struct starpu_profiling_task_info *);
		}
		else if (arg_type==STARPU_TASK_NO_SUBMITORDER)
		{
			(void)va_arg(varg_list, unsigned);
		}
		else if (arg_type==STARPU_TASK_SCHED_DATA)
		{
			(void)va_arg(varg_list, void *);
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d, did you perhaps forget to end arguments with 0?\n", arg_type);
		}
	}

	starpu_codelet_pack_arg_fini(&state, arg_buffer, arg_buffer_size);
	return 0;
}

void starpu_task_insert_data_make_room(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int current_buffer, int room)
{
	if (current_buffer + room > STARPU_NMAXBUFS)
	{
		if (*allocated_buffers == 0)
		{
			int i;
			struct starpu_codelet *cl2 = task->cl;
			*allocated_buffers = (current_buffer + room) * 2;
			_STARPU_MALLOC(task->dyn_handles, *allocated_buffers * sizeof(starpu_data_handle_t));
			for(i=0 ; i<current_buffer ; i++)
			{
				task->dyn_handles[i] = task->handles[i];
			}
			if (cl2->nbuffers == STARPU_VARIABLE_NBUFFERS || !cl2->dyn_modes)
			{
				_STARPU_MALLOC(task->dyn_modes, *allocated_buffers * sizeof(enum starpu_data_access_mode));
				for(i=0 ; i<current_buffer ; i++)
				{
					task->dyn_modes[i] = task->modes[i];
				}
			}
		}
		else if (current_buffer + room > *allocated_buffers)
		{
			*allocated_buffers = (current_buffer + room) * 2;
			_STARPU_REALLOC(task->dyn_handles, *allocated_buffers * sizeof(starpu_data_handle_t));
			if (cl->nbuffers == STARPU_VARIABLE_NBUFFERS || !cl->dyn_modes)
			{
				_STARPU_REALLOC(task->dyn_modes, *allocated_buffers * sizeof(enum starpu_data_access_mode));
			}
		}
	}
}

void starpu_task_insert_data_process_arg(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int *current_buffer, int arg_type, starpu_data_handle_t handle)
{
	enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type & ~STARPU_SSEND;
	STARPU_ASSERT(cl != NULL);
	STARPU_ASSERT_MSG(cl->nbuffers == STARPU_VARIABLE_NBUFFERS || *current_buffer < cl->nbuffers, "Too many data passed to starpu_task_insert");

	starpu_task_insert_data_make_room(cl, task, allocated_buffers, *current_buffer, 1);

	STARPU_TASK_SET_HANDLE(task, handle, *current_buffer);
	if (cl->nbuffers == STARPU_VARIABLE_NBUFFERS || (cl->nbuffers > STARPU_NMAXBUFS && !cl->dyn_modes))
		STARPU_TASK_SET_MODE(task, mode,* current_buffer);
	else if (STARPU_CODELET_GET_MODE(cl, *current_buffer))
	{
		STARPU_ASSERT_MSG(STARPU_CODELET_GET_MODE(cl, *current_buffer) == mode,
				"The codelet <%s> defines the access mode %d for the buffer %d which is different from the mode %d given to starpu_task_insert\n",
				cl->name, STARPU_CODELET_GET_MODE(cl, *current_buffer),
				*current_buffer, mode);
	}
	else
	{
#ifdef STARPU_DEVEL
#  warning shall we print a warning to the user
		/* Morse uses it to avoid having to set it in the codelet structure */
#endif
		STARPU_CODELET_SET_MODE(cl, mode, *current_buffer);
	}

	(*current_buffer)++;
}

void starpu_task_insert_data_process_array_arg(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int *current_buffer, int nb_handles, starpu_data_handle_t *handles)
{
	STARPU_ASSERT(cl != NULL);

	starpu_task_insert_data_make_room(cl, task, allocated_buffers, *current_buffer, nb_handles);

	int i;
	for(i=0 ; i<nb_handles ; i++)
	{
		STARPU_TASK_SET_HANDLE(task, handles[i], *current_buffer);
		(*current_buffer)++;
	}
}

void starpu_task_insert_data_process_mode_array_arg(struct starpu_codelet *cl, struct starpu_task *task, int *allocated_buffers, int *current_buffer, int nb_descrs, struct starpu_data_descr *descrs)
{
	STARPU_ASSERT(cl != NULL);

	starpu_task_insert_data_make_room(cl, task, allocated_buffers, *current_buffer, nb_descrs);

	int i;
	for(i=0 ; i<nb_descrs ; i++)
	{
		STARPU_ASSERT_MSG(cl->nbuffers == STARPU_VARIABLE_NBUFFERS || *current_buffer < cl->nbuffers, "Too many data passed to starpu_task_insert");
		STARPU_TASK_SET_HANDLE(task, descrs[i].handle, *current_buffer);
		if (task->dyn_modes)
		{
			task->dyn_modes[*current_buffer] = descrs[i].mode;
		}
		else if (cl->nbuffers == STARPU_VARIABLE_NBUFFERS || (cl->nbuffers > STARPU_NMAXBUFS && !cl->dyn_modes))
			STARPU_TASK_SET_MODE(task, descrs[i].mode, *current_buffer);
		else if (STARPU_CODELET_GET_MODE(cl, *current_buffer))
		{
			STARPU_ASSERT_MSG(STARPU_CODELET_GET_MODE(cl, *current_buffer) == descrs[i].mode,
					"The codelet <%s> defines the access mode %d for the buffer %d which is different from the mode %d given to starpu_task_insert\n",
					cl->name, STARPU_CODELET_GET_MODE(cl, *current_buffer),
					*current_buffer, descrs[i].mode);
		}
		else
		{
			STARPU_CODELET_SET_MODE(cl, descrs[i].mode, *current_buffer);
		}

		(*current_buffer)++;
	}

}

int _starpu_task_insert_create(struct starpu_codelet *cl, struct starpu_task *task, va_list varg_list)
{
	int arg_type;
	int current_buffer;
	int allocated_buffers = 0;
	unsigned ndeps = 0;
	unsigned nend_deps = 0;
	struct starpu_task **task_deps_array = NULL;
	struct starpu_task **task_end_deps_array = NULL;

	_STARPU_TRACE_TASK_BUILD_START();

	task->cl = cl;
	current_buffer = 0;

	struct starpu_codelet_pack_arg_data state;
	starpu_codelet_pack_arg_init(&state);

	while((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type & STARPU_R || arg_type & STARPU_W || arg_type & STARPU_SCRATCH || arg_type & STARPU_REDUX)
		{
			/* We have an access mode : we expect to find a handle */
			starpu_data_handle_t handle = va_arg(varg_list, starpu_data_handle_t);
			starpu_task_insert_data_process_arg(cl, task, &allocated_buffers, &current_buffer, arg_type, handle);
		}
		else if (arg_type == STARPU_DATA_ARRAY)
		{
			// Expect to find a array of handles and its size
			starpu_data_handle_t *handles = va_arg(varg_list, starpu_data_handle_t *);
			int nb_handles = va_arg(varg_list, int);
			starpu_task_insert_data_process_array_arg(cl, task, &allocated_buffers, &current_buffer, nb_handles, handles);
		}
		else if (arg_type==STARPU_DATA_MODE_ARRAY)
		{
			// Expect to find a array of descr and its size
			struct starpu_data_descr *descrs = va_arg(varg_list, struct starpu_data_descr *);
			int nb_descrs = va_arg(varg_list, int);
			starpu_task_insert_data_process_mode_array_arg(cl, task, &allocated_buffers, &current_buffer, nb_descrs, descrs);
		}
		else if (arg_type==STARPU_VALUE)
		{
			void *ptr = va_arg(varg_list, void *);
			size_t ptr_size = va_arg(varg_list, size_t);
			starpu_codelet_pack_arg(&state, ptr, ptr_size);
		}
		else if (arg_type==STARPU_CL_ARGS)
		{
			task->cl_arg = va_arg(varg_list, void *);
			task->cl_arg_size = va_arg(varg_list, size_t);
			task->cl_arg_free = 1;
		}
		else if (arg_type==STARPU_CL_ARGS_NFREE)
		{
			task->cl_arg = va_arg(varg_list, void *);
			task->cl_arg_size = va_arg(varg_list, size_t);
			task->cl_arg_free = 0;
		}
		else if (arg_type==STARPU_TASK_DEPS_ARRAY)
		{
			STARPU_ASSERT_MSG(task_deps_array == NULL, "Parameter 'STARPU_TASK_DEPS_ARRAY' passed twice not supported yet");
			ndeps = va_arg(varg_list, unsigned);
			task_deps_array = va_arg(varg_list, struct starpu_task **);
		}
		else if (arg_type==STARPU_TASK_END_DEPS_ARRAY)
		{
			STARPU_ASSERT_MSG(task_end_deps_array == NULL, "Parameter 'STARPU_TASK_END_DEPS_ARRAY' passed twice not supported yet");
			nend_deps = va_arg(varg_list, unsigned);
			task_end_deps_array = va_arg(varg_list, struct starpu_task **);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			task->callback_func = va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			task->callback_func = va_arg(varg_list, _starpu_callback_func_t);
			task->callback_arg = va_arg(varg_list, void *);
			task->callback_arg_free = 1;
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG_NFREE)
		{
			task->callback_func = va_arg(varg_list, _starpu_callback_func_t);
			task->callback_arg = va_arg(varg_list, void *);
			task->callback_arg_free = 0;
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			task->callback_arg = va_arg(varg_list, void *);
			task->callback_arg_free = 1;
		}
		else if (arg_type==STARPU_CALLBACK_ARG_NFREE)
		{
			task->callback_arg = va_arg(varg_list, void *);
			task->callback_arg_free = 0;
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
		{
			task->prologue_callback_func = va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
		{
			task->prologue_callback_arg = va_arg(varg_list, void *);
			task->prologue_callback_arg_free = 1;
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG_NFREE)
		{
			task->prologue_callback_arg = va_arg(varg_list, void *);
			task->prologue_callback_arg_free = 0;
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP)
		{
			task->prologue_callback_pop_func = va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG)
		{
			task->prologue_callback_pop_arg = va_arg(varg_list, void *);
			task->prologue_callback_pop_arg_free = 1;
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE)
		{
			task->prologue_callback_pop_arg = va_arg(varg_list, void *);
			task->prologue_callback_pop_arg_free = 0;
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			/* Followed by a priority level */
			int prio = va_arg(varg_list, int);
			task->priority = prio;
		}
		else if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_DATA)
		{
			(void)va_arg(varg_list, starpu_data_handle_t);
		}
		else if (arg_type==STARPU_EXECUTE_WHERE)
		{
			task->where = va_arg(varg_list, unsigned long long);
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			int worker = va_arg(varg_list, int);
			if (worker != -1)
			{
				task->workerid = worker;
				task->execute_on_a_specific_worker = 1;
			}
		}
		else if (arg_type==STARPU_WORKER_ORDER)
		{
			unsigned order = va_arg(varg_list, unsigned);
			if (order != 0)
			{
				STARPU_ASSERT_MSG(task->execute_on_a_specific_worker, "worker order only makes sense if a workerid is provided");
				task->workerorder = order;
			}
		}
		else if (arg_type==STARPU_SCHED_CTX)
		{
			unsigned sched_ctx = va_arg(varg_list, unsigned);
			task->sched_ctx = sched_ctx;
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			int hypervisor_tag = va_arg(varg_list, int);
			task->hypervisor_tag = hypervisor_tag;
		}
		else if (arg_type==STARPU_POSSIBLY_PARALLEL)
		{
			unsigned possibly_parallel = va_arg(varg_list, unsigned);
			task->possibly_parallel = possibly_parallel;
		}
		else if (arg_type==STARPU_FLOPS)
		{
			double flops = va_arg(varg_list, double);
			task->flops = flops;
		}
		else if (arg_type==STARPU_TAG)
		{
			starpu_tag_t tag = va_arg(varg_list, starpu_tag_t);
			task->tag_id = tag;
			task->use_tag = 1;
		}
		else if (arg_type==STARPU_TAG_ONLY)
		{
			starpu_tag_t tag = va_arg(varg_list, starpu_tag_t);
			task->tag_id = tag;
		}
		else if (arg_type==STARPU_NAME)
		{
			const char *name = va_arg(varg_list, const char *);
			task->name = name;
		}
		else if (arg_type==STARPU_NODE_SELECTION_POLICY)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_TASK_COLOR)
                {
                        task->color = va_arg(varg_list, int);
                }
		else if (arg_type==STARPU_TASK_SYNCHRONOUS)
                {
                        task->synchronous = va_arg(varg_list, int);
                }
		else if (arg_type==STARPU_HANDLES_SEQUENTIAL_CONSISTENCY)
                {
                        task->handles_sequential_consistency = va_arg(varg_list, unsigned char *);
                }
		else if (arg_type==STARPU_TASK_END_DEP)
		{
			int end_dep = va_arg(varg_list, int);
			starpu_task_end_dep_add(task, end_dep);
		}
		else if (arg_type==STARPU_TASK_WORKERIDS)
		{
			task->workerids_len = va_arg(varg_list, unsigned);
			task->workerids = va_arg(varg_list, uint32_t*);
		}
		else if (arg_type==STARPU_SEQUENTIAL_CONSISTENCY)
		{
			task->sequential_consistency = va_arg(varg_list, unsigned);
		}
		else if (arg_type==STARPU_TASK_PROFILING_INFO)
		{
			task->profiling_info = va_arg(varg_list, struct starpu_profiling_task_info *);
		}
		else if (arg_type==STARPU_TASK_NO_SUBMITORDER)
		{
			task->no_submitorder = va_arg(varg_list, unsigned);
		}
		else if (arg_type==STARPU_TASK_SCHED_DATA)
		{
			task->sched_data = va_arg(varg_list, void *);
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d, did you perhaps forget to end arguments with 0?\n", arg_type);
		}
	}

	if (cl)
	{
		if (cl->nbuffers == STARPU_VARIABLE_NBUFFERS)
		{
			task->nbuffers = current_buffer;
		}
		else
		{
			STARPU_ASSERT_MSG(current_buffer == cl->nbuffers, "Incoherent number of buffers between cl (%d) and number of parameters (%d)", cl->nbuffers, current_buffer);
		}
	}

	if (state.nargs)
	{
		if (task->cl_arg != NULL)
		{
			_STARPU_DISP("Parameters STARPU_CL_ARGS and STARPU_VALUE cannot be used in the same call\n");
			free(state.arg_buffer);
			return -EINVAL;
		}
		starpu_codelet_pack_arg_fini(&state, &task->cl_arg, &task->cl_arg_size);
	}

	if (task_deps_array)
	{
		starpu_task_declare_deps_array(task, ndeps, task_deps_array);
	}

	if (task_end_deps_array)
	{
		starpu_task_declare_end_deps_array(task, nend_deps, task_end_deps_array);
	}

	_STARPU_TRACE_TASK_BUILD_END();
	return 0;
}

int _fstarpu_task_insert_create(struct starpu_codelet *cl, struct starpu_task *task, void **arglist)
{
	int arg_i = 0;
	int current_buffer = 0;
	int allocated_buffers = 0;
	unsigned ndeps = 0;
	unsigned nend_deps = 0;
	struct starpu_task **task_deps_array = NULL;
	struct starpu_task **task_end_deps_array = NULL;

	_STARPU_TRACE_TASK_BUILD_START();

	struct starpu_codelet_pack_arg_data state;
	starpu_codelet_pack_arg_init(&state);

	task->cl = cl;
	task->name = NULL;
	task->cl_arg_free = 1;
	while (arglist[arg_i] != NULL)
	{
		const int arg_type = (int)(intptr_t)arglist[arg_i];
		if (arg_type & STARPU_R
			|| arg_type & STARPU_W
			|| arg_type & STARPU_SCRATCH
			|| arg_type & STARPU_REDUX)
		{
			arg_i++;
			starpu_data_handle_t handle = arglist[arg_i];
			starpu_task_insert_data_process_arg(cl, task, &allocated_buffers, &current_buffer, arg_type, handle);
		}
		else if (arg_type == STARPU_DATA_ARRAY)
		{
			arg_i++;
			starpu_data_handle_t *handles = arglist[arg_i];
			arg_i++;
			int nb_handles = *(int *)arglist[arg_i];
			starpu_task_insert_data_process_array_arg(cl, task, &allocated_buffers, &current_buffer, nb_handles, handles);
		}
		else if (arg_type == STARPU_DATA_MODE_ARRAY)
		{
			arg_i++;
			struct starpu_data_descr *descrs = arglist[arg_i];
			arg_i++;
			int nb_descrs = *(int *)arglist[arg_i];
			starpu_task_insert_data_process_mode_array_arg(cl, task, &allocated_buffers, &current_buffer, nb_descrs, descrs);
		}
		else if (arg_type == STARPU_VALUE)
		{
			arg_i++;
			void *ptr = arglist[arg_i];
			arg_i++;
			size_t ptr_size = (size_t)(intptr_t)arglist[arg_i];
			starpu_codelet_pack_arg(&state, ptr, ptr_size);
		}
		else if (arg_type == STARPU_CL_ARGS)
		{
			arg_i++;
			task->cl_arg = arglist[arg_i];
			arg_i++;
			task->cl_arg_size = (size_t)(intptr_t)arglist[arg_i];
			task->cl_arg_free = 1;
		}
		else if (arg_type == STARPU_CL_ARGS_NFREE)
		{
			arg_i++;
			task->cl_arg = arglist[arg_i];
			arg_i++;
			task->cl_arg_size = (size_t)(intptr_t)arglist[arg_i];
			task->cl_arg_free = 0;
		}
		else if (arg_type==STARPU_TASK_DEPS_ARRAY)
		{
			STARPU_ASSERT_MSG(task_deps_array == NULL, "Parameter 'STARPU_TASK_DEPS_ARRAY' passed twice not supported yet");
			arg_i++;
			ndeps = *(unsigned *)arglist[arg_i];
			arg_i++;
			task_deps_array = arglist[arg_i];
		}
		else if (arg_type==STARPU_TASK_END_DEPS_ARRAY)
		{
			STARPU_ASSERT_MSG(task_end_deps_array == NULL, "Parameter 'STARPU_TASK_END_DEPS_ARRAY' passed twice not supported yet");
			arg_i++;
			nend_deps = *(unsigned *)arglist[arg_i];
			arg_i++;
			task_end_deps_array = arglist[arg_i];
		}
		else if (arg_type == STARPU_CALLBACK)
		{
			arg_i++;
			task->callback_func = (_starpu_callback_func_t)arglist[arg_i];
		}
		else if (arg_type == STARPU_CALLBACK_WITH_ARG)
		{
			arg_i++;
			task->callback_func = (_starpu_callback_func_t)arglist[arg_i];
			arg_i++;
			task->callback_arg = arglist[arg_i];
			task->callback_arg_free = 1;
		}
		else if (arg_type == STARPU_CALLBACK_WITH_ARG_NFREE)
		{
			arg_i++;
			task->callback_func = (_starpu_callback_func_t)arglist[arg_i];
			arg_i++;
			task->callback_arg = arglist[arg_i];
			task->callback_arg_free = 0;
		}
		else if (arg_type == STARPU_CALLBACK_ARG)
		{
			arg_i++;
			task->callback_arg = arglist[arg_i];
			task->callback_arg_free = 1;
		}
		else if (arg_type == STARPU_CALLBACK_ARG_NFREE)
		{
			arg_i++;
			task->callback_arg = arglist[arg_i];
			task->callback_arg_free = 0;
		}
		else if (arg_type == STARPU_PROLOGUE_CALLBACK)
		{
			arg_i++;
			task->prologue_callback_func = (_starpu_callback_func_t)arglist[arg_i];
		}
		else if (arg_type == STARPU_PROLOGUE_CALLBACK_ARG)
		{
			arg_i++;
			task->prologue_callback_arg = arglist[arg_i];
			task->prologue_callback_arg_free = 1;
		}
		else if (arg_type == STARPU_PROLOGUE_CALLBACK_ARG_NFREE)
		{
			arg_i++;
			task->prologue_callback_arg = arglist[arg_i];
			task->prologue_callback_arg_free = 0;
		}
		else if (arg_type == STARPU_PROLOGUE_CALLBACK_POP)
		{
			arg_i++;
			task->prologue_callback_pop_func = (_starpu_callback_func_t)arglist[arg_i];
		}
		else if (arg_type == STARPU_PROLOGUE_CALLBACK_POP_ARG)
		{
			arg_i++;
			task->prologue_callback_pop_arg = arglist[arg_i];
			task->prologue_callback_pop_arg_free = 1;
		}
		else if (arg_type == STARPU_PROLOGUE_CALLBACK_POP_ARG_NFREE)
		{
			arg_i++;
			task->prologue_callback_pop_arg = arglist[arg_i];
			task->prologue_callback_pop_arg_free = 0;
		}
		else if (arg_type == STARPU_PRIORITY)
		{
			arg_i++;
			task->priority = *(int *)arglist[arg_i];
		}
		else if (arg_type == STARPU_EXECUTE_ON_NODE)
		{
			arg_i++;
			(void)arglist[arg_i];
		}
		else if (arg_type == STARPU_EXECUTE_ON_DATA)
		{
			arg_i++;
			(void)arglist[arg_i];
		}
		else if (arg_type == STARPU_EXECUTE_WHERE)
		{
			assert(0);
			arg_i++;
			unsigned long long where = *(unsigned long long *)arglist[arg_i];
			task->where = where;
		}
		else if (arg_type == STARPU_EXECUTE_ON_WORKER)
		{
			arg_i++;
			int worker = *(int *)arglist[arg_i];
			if (worker != -1)
			{
				task->workerid = worker;
				task->execute_on_a_specific_worker = 1;
			}
		}
		else if (arg_type == STARPU_WORKER_ORDER)
		{
			arg_i++;
			unsigned order = *(unsigned *)arglist[arg_i];
			if (order != 0)
			{
				STARPU_ASSERT_MSG(task->execute_on_a_specific_worker, "worker order only makes sense if a workerid is provided");
				task->workerorder = order;
			}
		}
		else if (arg_type == STARPU_SCHED_CTX)
		{
			arg_i++;
			task->sched_ctx = *(unsigned *)arglist[arg_i];
		}
		else if (arg_type == STARPU_HYPERVISOR_TAG)
		{
			arg_i++;
			task->hypervisor_tag = *(int *)arglist[arg_i];
		}
		else if (arg_type == STARPU_POSSIBLY_PARALLEL)
		{
			arg_i++;
			task->possibly_parallel = *(unsigned *)arglist[arg_i];
		}
		else if (arg_type == STARPU_FLOPS)
		{
			arg_i++;
			task->flops = *(double *)arglist[arg_i];
		}
		else if (arg_type == STARPU_TAG)
		{
			arg_i++;
			task->tag_id = *(starpu_tag_t *)arglist[arg_i];
			task->use_tag = 1;
		}
		else if (arg_type == STARPU_TAG_ONLY)
		{
			arg_i++;
			task->tag_id = *(starpu_tag_t *)arglist[arg_i];
		}
		else if (arg_type == STARPU_NAME)
		{
			arg_i++;
			task->name = arglist[arg_i];
		}
		else if (arg_type == STARPU_NODE_SELECTION_POLICY)
		{
			arg_i++;
			(void)arglist[arg_i];
		}
		else if (arg_type == STARPU_TASK_COLOR)
                {
                        arg_i++;
                        task->color = *(int *)arglist[arg_i];
                }
		else if (arg_type == STARPU_TASK_SYNCHRONOUS)
                {
                        arg_i++;
                        task->synchronous = *(int *)arglist[arg_i];
                }
		else if (arg_type==STARPU_HANDLES_SEQUENTIAL_CONSISTENCY)
                {
                        task->handles_sequential_consistency = (unsigned char *)arglist[arg_i];
                }
		else if (arg_type==STARPU_TASK_END_DEP)
		{
			arg_i++;
			starpu_task_end_dep_add(task, *(int*)arglist[arg_i]);
		}
		else if (arg_type==STARPU_TASK_WORKERIDS)
		{
			arg_i++;
			task->workerids_len = *(unsigned *)arglist[arg_i];
			arg_i++;
			task->workerids = (uint32_t *)arglist[arg_i];
		}
		else if (arg_type==STARPU_SEQUENTIAL_CONSISTENCY)
		{
			arg_i++;
			task->sequential_consistency = *(unsigned *)arglist[arg_i];
		}
		else if (arg_type==STARPU_TASK_PROFILING_INFO)
		{
			arg_i++;
			task->profiling_info = (struct starpu_profiling_task_info *)arglist[arg_i];
		}
		else if (arg_type==STARPU_TASK_NO_SUBMITORDER)
		{
			arg_i++;
			task->no_submitorder = *(unsigned *)arglist[arg_i];
		}
		else if (arg_type == STARPU_TASK_SCHED_DATA)
		{
			arg_i++;
			task->sched_data = (void*)arglist[arg_i];
		}
		else
		{
			STARPU_ABORT_MSG("unknown/unsupported argument %d, did you perhaps forget to end arguments with 0?", arg_type);
		}
		arg_i++;
	}

	if (cl)
	{
		if (cl->nbuffers == STARPU_VARIABLE_NBUFFERS)
		{
			task->nbuffers = current_buffer;
		}
		else
		{
			STARPU_ASSERT_MSG(current_buffer == cl->nbuffers, "Incoherent number of buffers between cl (%d) and number of parameters (%d)", cl->nbuffers, current_buffer);
		}
	}

	if (state.nargs)
	{
		if (task->cl_arg != NULL)
		{
			_STARPU_DISP("Parameters STARPU_CL_ARGS and STARPU_VALUE cannot be used in the same call\n");
			free(state.arg_buffer);
			return -EINVAL;
		}
		starpu_codelet_pack_arg_fini(&state, &task->cl_arg, &task->cl_arg_size);
	}

	if (task_deps_array)
	{
		starpu_task_declare_deps_array(task, ndeps, task_deps_array);
	}

	if (task_end_deps_array)
	{
		starpu_task_declare_end_deps_array(task, nend_deps, task_end_deps_array);
	}

	_STARPU_TRACE_TASK_BUILD_END();

	return 0;
}

/* Fortran interface to task_insert */
void fstarpu_task_insert(void **arglist)
{
	struct starpu_codelet *cl = arglist[0];
	if (cl == NULL)
	{
		STARPU_ABORT_MSG("task without codelet");
	}
	struct starpu_task *task = starpu_task_create();
	int ret = _fstarpu_task_insert_create(cl, task, arglist+1);
	if (ret != 0)
	{
		STARPU_ABORT_MSG("task creation failed");
	}
	ret = starpu_task_submit(task);
	if (ret != 0)
	{
		STARPU_ABORT_MSG("starpu_task_submit failed");
	}
}

/* fstarpu_insert_task: aliased to fstarpu_task_insert in fstarpu_mod.f90 */
