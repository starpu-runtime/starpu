/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011, 2013-2015   Université Bordeaux
 * Copyright (C) 2011-2015         CNRS
 * Copyright (C) 2011, 2014        INRIA
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

/* Deal with callbacks. The unpack function may be called multiple times when
 * we have a parallel task, and we should not free the cl_arg parameter from
 * the callback function. */
struct _starpu_task_insert_cb_wrapper
{
	_starpu_callback_func_t callback_func;
	void *callback_arg;
};

static
void _starpu_task_insert_callback_wrapper(void *_cl_arg_wrapper)
{
	struct _starpu_task_insert_cb_wrapper *cl_arg_wrapper = (struct _starpu_task_insert_cb_wrapper *) _cl_arg_wrapper;

	/* Execute the callback specified by the application */
	if (cl_arg_wrapper->callback_func)
		cl_arg_wrapper->callback_func(cl_arg_wrapper->callback_arg);
}

static
void _starpu_pack_arguments(size_t *current_offset, size_t *arg_buffer_size_, char **arg_buffer_, void *ptr, size_t ptr_size)
{
	if (*current_offset + sizeof(ptr_size) + ptr_size > *arg_buffer_size_)
	{
		if (*arg_buffer_size_ == 0)
			*arg_buffer_size_ = 128 + sizeof(ptr_size) + ptr_size;
		else
			*arg_buffer_size_ = 2 * *arg_buffer_size_ + sizeof(ptr_size) + ptr_size;
		*arg_buffer_ = realloc(*arg_buffer_, *arg_buffer_size_);
	}
	memcpy(*arg_buffer_+*current_offset, (void *)&ptr_size, sizeof(ptr_size));
	*current_offset += sizeof(ptr_size);

	memcpy(*arg_buffer_+*current_offset, ptr, ptr_size);
	*current_offset += ptr_size;
	STARPU_ASSERT(*current_offset <= *arg_buffer_size_);
}

int _starpu_codelet_pack_args(void **arg_buffer, size_t *arg_buffer_size, va_list varg_list)
{
	int arg_type;
	int nargs = 0;
	char *_arg_buffer = NULL; // We would like a void* but we use a char* to allow pointer arithmetic
	size_t _arg_buffer_size = 0;
	size_t current_offset = sizeof(nargs);

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

			nargs++;
			_starpu_pack_arguments(&current_offset, &_arg_buffer_size, &_arg_buffer, ptr, ptr_size);
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
		else if (arg_type==STARPU_CALLBACK_ARG)
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
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP)
		{
			va_arg(varg_list, _starpu_callback_func_t);
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG)
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
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			va_arg(varg_list, int);
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
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}
	}

	if (nargs)
	{
		memcpy(_arg_buffer, (int *)&nargs, sizeof(nargs));
	}
	else
	{
		free(_arg_buffer);
		_arg_buffer = NULL;
	}

	*arg_buffer = _arg_buffer;
	*arg_buffer_size = _arg_buffer_size;
	return 0;
}

static
void _starpu_task_insert_check_nb_buffers(struct starpu_codelet *cl, struct starpu_task **task, int *allocated_buffers, int current_buffer)
{
	if (current_buffer >= STARPU_NMAXBUFS)
	{
		if (*allocated_buffers == 0)
		{
			int i;
			struct starpu_codelet *cl2 = (*task)->cl;
			*allocated_buffers = STARPU_NMAXBUFS * 2;
			(*task)->dyn_handles = malloc(*allocated_buffers * sizeof(starpu_data_handle_t));
			for(i=0 ; i<current_buffer ; i++)
			{
				(*task)->dyn_handles[i] = (*task)->handles[i];
			}
			if (cl2->nbuffers == STARPU_VARIABLE_NBUFFERS || !cl2->dyn_modes)
			{
				(*task)->dyn_modes = malloc(*allocated_buffers * sizeof(enum starpu_data_access_mode));
				for(i=0 ; i<current_buffer ; i++)
				{
					(*task)->dyn_modes[i] = (*task)->modes[i];
				}
			}
		}
		else if (current_buffer >= *allocated_buffers)
		{
			*allocated_buffers *= 2;
			(*task)->dyn_handles = realloc((*task)->dyn_handles, *allocated_buffers * sizeof(starpu_data_handle_t));
			if (cl->nbuffers == STARPU_VARIABLE_NBUFFERS || !cl->dyn_modes)
				(*task)->dyn_modes = realloc((*task)->dyn_modes, *allocated_buffers * sizeof(enum starpu_data_access_mode));
		}
	}
}

void _starpu_task_insert_create(struct starpu_codelet *cl, struct starpu_task **task, va_list varg_list)
{
	int arg_type;
	char *arg_buffer_ = NULL;
	size_t arg_buffer_size_ = 0;
	size_t current_offset = sizeof(int);
	int current_buffer;
	int nargs = 0;
	int allocated_buffers = 0;

	struct _starpu_task_insert_cb_wrapper *cl_arg_wrapper = (struct _starpu_task_insert_cb_wrapper *) malloc(sizeof(struct _starpu_task_insert_cb_wrapper));
	STARPU_ASSERT(cl_arg_wrapper);

	cl_arg_wrapper->callback_func = NULL;

	struct _starpu_task_insert_cb_wrapper *prologue_cl_arg_wrapper = (struct _starpu_task_insert_cb_wrapper *) malloc(sizeof(struct _starpu_task_insert_cb_wrapper));
	STARPU_ASSERT(prologue_cl_arg_wrapper);

	prologue_cl_arg_wrapper->callback_func = NULL;

	struct _starpu_task_insert_cb_wrapper *prologue_pop_cl_arg_wrapper = (struct _starpu_task_insert_cb_wrapper *) malloc(sizeof(struct _starpu_task_insert_cb_wrapper));
	STARPU_ASSERT(prologue_pop_cl_arg_wrapper);

	prologue_pop_cl_arg_wrapper->callback_func = NULL;

	(*task)->cl = cl;
	current_buffer = 0;

	while((arg_type = va_arg(varg_list, int)) != 0)
	{
		if (arg_type & STARPU_R || arg_type & STARPU_W || arg_type & STARPU_SCRATCH || arg_type & STARPU_REDUX)
		{
			/* We have an access mode : we expect to find a handle */
			starpu_data_handle_t handle = va_arg(varg_list, starpu_data_handle_t);
			enum starpu_data_access_mode mode = (enum starpu_data_access_mode) arg_type & ~STARPU_SSEND;

			STARPU_ASSERT(cl != NULL);

			_starpu_task_insert_check_nb_buffers(cl, task, &allocated_buffers, current_buffer);

			STARPU_TASK_SET_HANDLE((*task), handle, current_buffer);
			if (cl->nbuffers == STARPU_VARIABLE_NBUFFERS || (cl->nbuffers > STARPU_NMAXBUFS && !cl->dyn_modes))
				STARPU_TASK_SET_MODE(*task, mode, current_buffer);
			else if (STARPU_CODELET_GET_MODE(cl, current_buffer))
			{
				STARPU_ASSERT_MSG(STARPU_CODELET_GET_MODE(cl, current_buffer) == mode,
						   "The codelet <%s> defines the access mode %d for the buffer %d which is different from the mode %d given to starpu_task_insert\n",
						  cl->name, STARPU_CODELET_GET_MODE(cl, current_buffer),
						  current_buffer, mode);
			}
			else
			{
#ifdef STARPU_DEVEL
#  warning shall we print a warning to the user
/* Morse uses it to avoid having to set it in the codelet structure */
#endif
				STARPU_CODELET_SET_MODE(cl, mode, current_buffer);
			}

			current_buffer++;
		}
		else if (arg_type == STARPU_DATA_ARRAY)
		{
			// Expect to find a array of handles and its size
			starpu_data_handle_t *handles = va_arg(varg_list, starpu_data_handle_t *);
			int nb_handles = va_arg(varg_list, int);

			STARPU_ASSERT(cl != NULL);

			int i;
			for(i=0 ; i<nb_handles ; i++)
			{
				_starpu_task_insert_check_nb_buffers(cl, task, &allocated_buffers, current_buffer);
				STARPU_TASK_SET_HANDLE((*task), handles[i], current_buffer);
				current_buffer++;
			}

		}
		else if (arg_type==STARPU_DATA_MODE_ARRAY)
		{
			// Expect to find a array of descr and its size
			struct starpu_data_descr *descrs = va_arg(varg_list, struct starpu_data_descr *);
			int nb_descrs = va_arg(varg_list, int);

			STARPU_ASSERT(cl != NULL);

			int i;
			for(i=0 ; i<nb_descrs ; i++)
			{
				_starpu_task_insert_check_nb_buffers(cl, task, &allocated_buffers, current_buffer);
				STARPU_TASK_SET_HANDLE((*task), descrs[i].handle, current_buffer);
				if ((*task)->dyn_modes)
				{
					(*task)->dyn_modes[i] = descrs[i].mode;
				}
				else if (cl->nbuffers == STARPU_VARIABLE_NBUFFERS || (cl->nbuffers > STARPU_NMAXBUFS && !cl->dyn_modes))
					STARPU_TASK_SET_MODE(*task, descrs[i].mode, current_buffer);
				else if (STARPU_CODELET_GET_MODE(cl, current_buffer))
				{
					STARPU_ASSERT_MSG(STARPU_CODELET_GET_MODE(cl, current_buffer) == descrs[i].mode,
							  "The codelet <%s> defines the access mode %d for the buffer %d which is different from the mode %d given to starpu_task_insert\n",
							  cl->name, STARPU_CODELET_GET_MODE(cl, current_buffer),
							  current_buffer, descrs[i].mode);
				}
				else
				{
					STARPU_CODELET_SET_MODE(cl, descrs[i].mode, current_buffer);
				}

				current_buffer++;
			}

		}
		else if (arg_type==STARPU_VALUE)
		{
			void *ptr = va_arg(varg_list, void *);
			size_t ptr_size = va_arg(varg_list, size_t);

			nargs++;
			_starpu_pack_arguments(&current_offset, &arg_buffer_size_, &arg_buffer_, ptr, ptr_size);
		}
		else if (arg_type==STARPU_CALLBACK)
		{
			void (*callback_func)(void *);
			callback_func = va_arg(varg_list, _starpu_callback_func_t);
			cl_arg_wrapper->callback_func = callback_func;
		}
		else if (arg_type==STARPU_CALLBACK_WITH_ARG)
		{
			void (*callback_func)(void *);
			void *callback_arg;
			callback_func = va_arg(varg_list, _starpu_callback_func_t);
			callback_arg = va_arg(varg_list, void *);
			cl_arg_wrapper->callback_func = callback_func;
			cl_arg_wrapper->callback_arg = callback_arg;
		}
		else if (arg_type==STARPU_CALLBACK_ARG)
		{
			void *callback_arg = va_arg(varg_list, void *);
			cl_arg_wrapper->callback_arg = callback_arg;
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK)
		{
			void (*callback_func)(void *);
			callback_func = va_arg(varg_list, _starpu_callback_func_t);
			prologue_cl_arg_wrapper->callback_func = callback_func;
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_ARG)
		{
			void *callback_arg = va_arg(varg_list, void *);
			prologue_cl_arg_wrapper->callback_arg = callback_arg;
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP)
		{
			void (*callback_func)(void *);
			callback_func = va_arg(varg_list, _starpu_callback_func_t);
			prologue_pop_cl_arg_wrapper->callback_func = callback_func;
		}
		else if (arg_type==STARPU_PROLOGUE_CALLBACK_POP_ARG)
		{
			void *callback_arg = va_arg(varg_list, void *);
			prologue_pop_cl_arg_wrapper->callback_arg = callback_arg;
		}
		else if (arg_type==STARPU_PRIORITY)
		{
			/* Followed by a priority level */
			int prio = va_arg(varg_list, int);
			(*task)->priority = prio;
		}
		else if (arg_type==STARPU_EXECUTE_ON_NODE)
		{
			(void)va_arg(varg_list, int);
		}
		else if (arg_type==STARPU_EXECUTE_ON_DATA)
		{
			(void)va_arg(varg_list, starpu_data_handle_t);
		}
		else if (arg_type==STARPU_EXECUTE_ON_WORKER)
		{
			int worker = va_arg(varg_list, int);
			if (worker != -1)
			{
				(*task)->workerid = worker;
				(*task)->execute_on_a_specific_worker = 1;
			}
		}
		else if (arg_type==STARPU_WORKER_ORDER)
		{
			unsigned order = va_arg(varg_list, unsigned);
			if (order != 0)
			{
				STARPU_ASSERT_MSG((*task)->execute_on_a_specific_worker, "worker order only makes sense if a workerid is provided");
				(*task)->workerorder = order;
			}
		}
		else if (arg_type==STARPU_SCHED_CTX)
		{
			unsigned sched_ctx = va_arg(varg_list, unsigned);
			(*task)->sched_ctx = sched_ctx;
		}
		else if (arg_type==STARPU_HYPERVISOR_TAG)
		{
			int hypervisor_tag = va_arg(varg_list, int);
			(*task)->hypervisor_tag = hypervisor_tag;
		}
		else if (arg_type==STARPU_POSSIBLY_PARALLEL)
		{
			unsigned possibly_parallel = va_arg(varg_list, unsigned);
			(*task)->possibly_parallel = possibly_parallel;
		}
		else if (arg_type==STARPU_FLOPS)
		{
			double flops = va_arg(varg_list, double);
			(*task)->flops = flops;
		}
		else if (arg_type==STARPU_TAG)
		{
			starpu_tag_t tag = va_arg(varg_list, starpu_tag_t);
			(*task)->tag_id = tag;
			(*task)->use_tag = 1;
		}
		else if (arg_type==STARPU_TAG_ONLY)
		{
			starpu_tag_t tag = va_arg(varg_list, starpu_tag_t);
			(*task)->tag_id = tag;
		}
		else if (arg_type==STARPU_NAME)
		{
			const char *name = va_arg(varg_list, const char *);
			(*task)->name = name;
		}
		else if (arg_type==STARPU_NODE_SELECTION_POLICY)
		{
			(void)va_arg(varg_list, int);
		}
		else
		{
			STARPU_ABORT_MSG("Unrecognized argument %d\n", arg_type);
		}
	}

	if (cl)
	{
		if (cl->nbuffers == STARPU_VARIABLE_NBUFFERS)
		{
			(*task)->nbuffers = current_buffer;
		}
		else
		{
			STARPU_ASSERT_MSG(current_buffer == cl->nbuffers, "Incoherent number of buffers between cl (%d) and number of parameters (%d)", cl->nbuffers, current_buffer);
		}
	}

	if (nargs)
	{
		memcpy(arg_buffer_, (int *)&nargs, sizeof(nargs));
		(*task)->cl_arg = arg_buffer_;
		(*task)->cl_arg_size = arg_buffer_size_;
	}
	else
	{
		free(arg_buffer_);
		arg_buffer_ = NULL;
	}

	/* The callback will free the argument stack and execute the
	 * application's callback, if any. */
	(*task)->callback_func = _starpu_task_insert_callback_wrapper;
	(*task)->callback_arg = cl_arg_wrapper;
	(*task)->callback_arg_free = 1;

	(*task)->prologue_callback_func = _starpu_task_insert_callback_wrapper;
	(*task)->prologue_callback_arg = prologue_cl_arg_wrapper;
	(*task)->prologue_callback_arg_free = 1;

	(*task)->prologue_callback_pop_func = _starpu_task_insert_callback_wrapper;
	(*task)->prologue_callback_pop_arg = prologue_pop_cl_arg_wrapper;
	(*task)->prologue_callback_pop_arg_free = 1;
}
