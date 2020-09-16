/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DEBUG_H__
#define __DEBUG_H__

/** @file */

#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>

#include <common/config.h>
#include <core/workers.h>

#if defined(STARPU_USE_AYUDAME1)
/* Ayudame 1 API */
# include <Ayudame.h>
# ifndef AYU_RT_STARPU
#  define AYU_RT_STARPU 4
# endif
# define STARPU_AYU_EVENT AYU_event

# define STARPU_AYU_PREINIT() \
	if (AYU_event) \
	{ \
		enum ayu_runtime_t ayu_rt = AYU_RT_STARPU; \
		AYU_event(AYU_PREINIT, 0, (void*) &ayu_rt); \
	}

# define STARPU_AYU_INIT() \
	if (AYU_event) \
	{ \
		AYU_event(AYU_INIT, 0, NULL); \
	}

# define STARPU_AYU_FINISH() \
	if (AYU_event) \
	{ \
		AYU_event(AYU_FINISH, 0, NULL); \
	}

# define STARPU_AYU_ADDDEPENDENCY(previous, handle, job_id) \
	if (AYU_event) \
	{ \
		uintptr_t __AYU_data[3] = { (previous), (uintptr_t) (handle), (uintptr_t) (handle) }; \
		AYU_event(AYU_ADDDEPENDENCY, (job_id), __AYU_data); \
	}

# define STARPU_AYU_REMOVETASK(job_id) \
	if (AYU_event) \
	{ \
		AYU_event(AYU_REMOVETASK, (job_id), NULL); \
	}

# define STARPU_AYU_ADDTASK(job_id, task) \
	if (AYU_event) \
	{ \
		int64_t __AYU_data[2] = { \
			((struct starpu_task *)(task))!=NULL?_starpu_ayudame_get_func_id(((struct starpu_task *)(task))->cl):0, \
			((struct starpu_task *)(task))!=NULL?((struct starpu_task *)(task))->priority-STARPU_MIN_PRIO:0 \
		}; \
		AYU_event(AYU_ADDTASK, (job_id), __AYU_data); \
	}

# define STARPU_AYU_PRERUNTASK(job_id, workerid) \
	if (AYU_event) \
	{ \
		intptr_t __id = (workerid); \
		AYU_event(AYU_PRERUNTASK, (job_id), &__id); \
	}

# define STARPU_AYU_RUNTASK(job_id) \
	if (AYU_event) \
	{ \
		AYU_event(AYU_RUNTASK, (job_id), NULL); \
	}

# define STARPU_AYU_POSTRUNTASK(job_id) \
	if (AYU_event) \
	{ \
		AYU_event(AYU_POSTRUNTASK, (job_id), NULL); \
	}

# define STARPU_AYU_ADDTOTASKQUEUE(job_id, worker_id) \
	if (AYU_event) \
	{ \
		intptr_t __id = (worker_id); \
		AYU_event(AYU_ADDTASKTOQUEUE, (job_id), &__id); \
	}

# define STARPU_AYU_BARRIER() \
	if (AYU_event) \
	{ \
		AYU_event(AYU_BARRIER, 0, NULL); \
	}

#elif defined(STARPU_USE_AYUDAME2)
/* Ayudame 2 API */
# include <ayudame.h>
# define STARPU_AYU_EVENT ayu_event

# define STARPU_AYU_PREINIT()

# define STARPU_AYU_INIT()

# define STARPU_AYU_FINISH() \
	if (ayu_event){ \
		ayu_client_id_t __cli_id = get_client_id(AYU_CLIENT_STARPU); \
		ayu_event_data_t __data; \
		__data.common.client_id = __cli_id; \
		ayu_event(AYU_FINISH, __data); \
	}

# define STARPU_AYU_ADDDEPENDENCY(previous, handle, job_id) \
	if (ayu_event) \
	{ \
		ayu_client_id_t __cli_id = get_client_id(AYU_CLIENT_STARPU); \
		ayu_event_data_t __data; \
		uint64_t __dep_id=0; \
		__dep_id |= (previous) << 0; \
		__dep_id |= (job_id) << 24; \
		__dep_id |= (uintptr_t) (handle) << 48; \
		__data.common.client_id = __cli_id; \
		__data.add_dependency.dependency_id = __dep_id; \
		__data.add_dependency.from_id=(previous); \
		__data.add_dependency.to_id=(job_id); \
		__data.add_dependency.dependency_label = "dep"; \
		ayu_event(AYU_ADDDEPENDENCY, __data); \
		ayu_wipe_data(&__data); \
		\
		char __buf[32]; \
		snprintf(__buf, sizeof(__buf), "%llu", (unsigned long long)(uintptr_t) (handle)); \
		__data.common.client_id = __cli_id; \
		__data.set_property.property_owner_id = __dep_id; \
		__data.set_property.key = "dep_address_value"; \
		__data.set_property.value = __buf; \
		ayu_event(AYU_SETPROPERTY, __data); \
		ayu_wipe_data(&__data); \
	}

# define STARPU_AYU_REMOVETASK(job_id) \
	if (ayu_event) \
	{ \
		ayu_client_id_t __cli_id = get_client_id(AYU_CLIENT_STARPU); \
		ayu_event_data_t __data; \
		__data.common.client_id = __cli_id; \
		__data.set_property.property_owner_id = (job_id); \
		__data.set_property.key = "state"; \
		__data.set_property.value = "finished"; \
		ayu_event(AYU_SETPROPERTY, __data); \
		ayu_wipe_data(&__data); \
	}

# define STARPU_AYU_ADDTASK(job_id, task) \
	if (ayu_event) \
	{ \
		ayu_client_id_t __cli_id = get_client_id(AYU_CLIENT_STARPU); \
		ayu_event_data_t __data; \
		__data.common.client_id = __cli_id; \
		__data.add_task.task_id = (job_id); \
		__data.add_task.scope_id = 0; \
		__data.add_task.task_label = "task"; \
		ayu_event(AYU_ADDTASK, __data); \
		ayu_wipe_data(&__data); \
		\
		if ((task) != NULL) \
		{ \
			char __buf[32]; \
			snprintf(__buf, sizeof(__buf), "%d", ((struct starpu_task *)(task))->priority); \
			__data.common.client_id = __cli_id; \
			__data.set_property.property_owner_id = (job_id); \
			__data.set_property.key = "priority"; \
			__data.set_property.value = __buf; \
			ayu_event(AYU_SETPROPERTY, __data); \
			ayu_wipe_data(&__data); \
			\
			const char *__name = ((struct starpu_task *)(task))->name != NULL?((struct starpu_task *)(task))->name: \
			             ((struct starpu_task *)(task))->cl->name != NULL?((struct starpu_task *)(task))->cl->name:"<no_name>"; \
			__data.common.client_id = __cli_id; \
			__data.set_property.property_owner_id = (job_id); \
			__data.set_property.key = "function_name"; \
			__data.set_property.value = __name; \
			ayu_event(AYU_SETPROPERTY, __data); \
			ayu_wipe_data(&__data); \
		} \
	}

# define STARPU_AYU_PRERUNTASK(job_id, workerid) \
	if (ayu_event) \
	{ \
		ayu_client_id_t __cli_id = get_client_id(AYU_CLIENT_STARPU); \
		ayu_event_data_t __data; \
		__data.common.client_id = __cli_id; \
		__data.set_property.property_owner_id = (job_id); \
		__data.set_property.key = "state"; \
		__data.set_property.value = "running"; \
		ayu_event(AYU_SETPROPERTY, __data); \
		ayu_wipe_data(&__data); \
		\
		char __buf[32]; \
		snprintf(__buf, sizeof(__buf), "%d", (workerid));	\
		__data.common.client_id = __cli_id; \
		__data.set_property.property_owner_id = (job_id); \
		__data.set_property.key = "worker"; \
		__data.set_property.value = __buf; \
		ayu_event(AYU_SETPROPERTY, __data); \
		ayu_wipe_data(&__data); \
	}

# define STARPU_AYU_RUNTASK(job_id) \
	if (ayu_event) { \
		ayu_client_id_t __cli_id = get_client_id(AYU_CLIENT_STARPU); \
		ayu_event_data_t __data; \
		__data.common.client_id = __cli_id; \
		__data.set_property.property_owner_id = (job_id); \
		__data.set_property.key = "state"; \
		__data.set_property.value = "running"; \
		ayu_event(AYU_SETPROPERTY, __data); \
		ayu_wipe_data(&__data); \
	}

# define STARPU_AYU_POSTRUNTASK(job_id) \
	if (ayu_event) \
	{ \
		/* TODO ADD thread id core id etc */ \
		ayu_client_id_t __cli_id = get_client_id(AYU_CLIENT_STARPU); \
		ayu_event_data_t __data; \
		__data.common.client_id = __cli_id; \
		__data.set_property.property_owner_id = (job_id); \
		__data.set_property.key = "state"; \
		__data.set_property.value = "finished"; \
		ayu_event(AYU_SETPROPERTY, __data); \
		ayu_wipe_data(&__data); \
	}

# define STARPU_AYU_ADDTOTASKQUEUE(job_id, worker_id) \
	if (ayu_event) \
	{ \
		ayu_client_id_t __cli_id = get_client_id(AYU_CLIENT_STARPU); \
		ayu_event_data_t __data; \
		__data.common.client_id = __cli_id; \
		__data.set_property.property_owner_id = (job_id); \
		__data.set_property.key = "state"; \
		__data.set_property.value = "queued"; \
		ayu_event(AYU_SETPROPERTY, __data); \
		ayu_wipe_data(&__data); \
		\
		char __buf[32]; \
		snprintf(__buf, sizeof(__buf), "%d", (int)(worker_id));	\
		__data.common.client_id = __cli_id; \
		__data.set_property.property_owner_id = (job_id); \
		__data.set_property.key = "worker"; \
		__data.set_property.value = __buf; \
		ayu_event(AYU_SETPROPERTY, __data); \
		ayu_wipe_data(&__data); \
 	}

# define STARPU_AYU_BARRIER() \
	if (ayu_event) \
	{ \
		/* How to generate a barrier event with Ayudame 2? */ \
	}
#else
# define STARPU_AYU_EVENT (0)
# define STARPU_AYU_PREINIT()
# define STARPU_AYU_INIT()
# define STARPU_AYU_FINISH()
# define STARPU_AYU_ADDDEPENDENCY(previous, handle, next_job)
# define STARPU_AYU_REMOVETASK(job_id)
# define STARPU_AYU_ADDTASK(job_id, task)
# define STARPU_AYU_PRERUNTASK(job_id, workerid)
# define STARPU_AYU_RUNTASK(job_id)
# define STARPU_AYU_POSTRUNTASK(job_id)
# define STARPU_AYU_ADDTOTASKQUEUE(job_id, worker_id)
# define STARPU_AYU_BARRIER()

#endif

/** Create a file that will contain StarPU's log */
void _starpu_open_debug_logfile(void);

/** Close StarPU's log file */
void _starpu_close_debug_logfile(void);

/** Write into StarPU's log file */
void _starpu_print_to_logfile(const char *format, ...) STARPU_ATTRIBUTE_FORMAT(printf, 1, 2);

/** Tell gdb whether FXT is compiled in or not */
extern int _starpu_use_fxt;

#if defined(STARPU_USE_AYUDAME1)
/** Get an Ayudame id for CL */
int64_t _starpu_ayudame_get_func_id(struct starpu_codelet *cl);
#endif

void _starpu_watchdog_init(void);
void _starpu_watchdog_shutdown(void);

#endif // __DEBUG_H__
