/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <common/utils.h>
#include <core/simgrid.h>
#include <starpu_util.h>
#include <starpu_profiling.h>
#include <core/workers.h>

/* we need to identify each task to generate the DAG. */
unsigned long _starpu_job_cnt = 0;

#ifdef STARPU_USE_FXT
#include <common/fxt.h>
#include <starpu_fxt.h>
#include <sys/stat.h>

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

#ifdef __linux__
#include <sys/syscall.h>   /* for SYS_gettid */
#elif defined(__FreeBSD__)
#include <sys/thr.h>       /* for thr_self() */
#endif

/* By default, record all events but the VERBOSE_EXTRA ones, which are very costly: */
#define KEYMASKALL_DEFAULT FUT_KEYMASKALL & (~_STARPU_FUT_KEYMASK_TASK_VERBOSE_EXTRA) & (~_STARPU_FUT_KEYMASK_MPI_VERBOSE_EXTRA)

static char _starpu_prof_file_user[128];
int _starpu_fxt_started = 0;
int _starpu_fxt_willstart = 1;
starpu_pthread_mutex_t _starpu_fxt_started_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
starpu_pthread_cond_t _starpu_fxt_started_cond = STARPU_PTHREAD_COND_INITIALIZER;

/* and their submission order. */
unsigned long _starpu_submit_order = 0;

static int _starpu_written = 0;

static int _starpu_id;

/* If we use several MPI processes, we can't use STARPU_GENERATE_TRACE=1,
 * because each MPI process will handle its own trace file, so store the world
 * size to warn the user if needed and avoid processing partial traces. */
static int _starpu_mpi_worldsize = 1;

/* Event mask used to initialize FxT. By default all events are recorded just
 * after FxT starts, but this can be changed by calling
 * starpu_fxt_autostart_profiling(0) */
static unsigned int initial_key_mask = KEYMASKALL_DEFAULT;

/* Event mask used when events are actually recorded, e.g. between
 * starpu_fxt_start|stop_profiling() calls if autostart is disabled, or at
 * anytime otherwise. Can be changed by the user at runtime, by setting
 * STARPU_FXT_EVENTS env var. */
static unsigned int profiling_key_mask = 0;

#ifdef STARPU_SIMGRID
/* Give virtual time to FxT */
uint64_t fut_getstamp(void)
{
	return starpu_timing_now()*1000.;
}
#endif

long _starpu_gettid(void)
{
	/* TODO: test at configure whether __thread is available, and use that
	 * to cache the value.
	 * Don't use the TSD, this is getting called before we would have the
	 * time to allocate it.  */
#ifdef STARPU_SIMGRID
#  ifdef HAVE_SG_ACTOR_SELF
	return (uintptr_t) sg_actor_self();
#  else
	return (uintptr_t) MSG_process_self();
#  endif
#else
#if defined(__linux__)
	return syscall(SYS_gettid);
#elif defined(__FreeBSD__)
	long tid;
	thr_self(&tid);
	return tid;
#elif defined(_WIN32) && !defined(__CYGWIN__)
	return (long) GetCurrentThreadId();
#else
	return (long) starpu_pthread_self();
#endif
#endif
}

static void _starpu_profile_set_tracefile(void)
{
	char *user;

	char *fxt_prefix = starpu_getenv("STARPU_FXT_PREFIX");
	if (!fxt_prefix)
		fxt_prefix = "/tmp";
	else
		_starpu_mkpath_and_check(fxt_prefix, S_IRWXU);

	char suffix[64];
	char *fxt_suffix = starpu_getenv("STARPU_FXT_SUFFIX");
	if (!fxt_suffix)
	{
		user = starpu_getenv("USER");
		if (!user)
			user = "";
		snprintf(suffix, sizeof(suffix), "prof_file_%s_%d", user, _starpu_id);
	}
	else
	{
		snprintf(suffix, sizeof(suffix), "%s_%d", fxt_suffix, _starpu_id);
	}

	snprintf(_starpu_prof_file_user, sizeof(_starpu_prof_file_user), "%s/%s", fxt_prefix, suffix);
}

static inline unsigned int _starpu_profile_get_user_keymask(void)
{
	if (profiling_key_mask != 0)
		return profiling_key_mask;

	char *fxt_events = starpu_getenv("STARPU_FXT_EVENTS");
	if (fxt_events)
	{
		profiling_key_mask = _STARPU_FUT_KEYMASK_META; // contains mandatory events, even when profiling is disabled

		char delim[] = "|,";
		char* sub = strtok(fxt_events, delim);
		for (; sub != NULL; sub = strtok(NULL, delim))
		{
			if (!strcasecmp(sub, "USER"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_USER;
			else if (!strcasecmp(sub, "TASK"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_TASK;
			else if (!strcasecmp(sub, "TASK_VERBOSE"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_TASK_VERBOSE;
			else if (!strcasecmp(sub, "DATA"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_DATA;
			else if (!strcasecmp(sub, "DATA_VERBOSE"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_DATA_VERBOSE;
			else if (!strcasecmp(sub, "WORKER"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_WORKER;
			else if (!strcasecmp(sub, "WORKER_VERBOSE"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_WORKER_VERBOSE;
			else if (!strcasecmp(sub, "DSM"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_DSM;
			else if (!strcasecmp(sub, "DSM_VERBOSE"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_DSM_VERBOSE;
			else if (!strcasecmp(sub, "SCHED"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_SCHED;
			else if (!strcasecmp(sub, "SCHED_VERBOSE"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_SCHED_VERBOSE;
			else if (!strcasecmp(sub, "LOCK"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_LOCK;
			else if (!strcasecmp(sub, "LOCK_VERBOSE"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_LOCK_VERBOSE;
			else if (!strcasecmp(sub, "EVENT"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_EVENT;
			else if (!strcasecmp(sub, "EVENT_VERBOSE"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_EVENT_VERBOSE;
			else if (!strcasecmp(sub, "MPI"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_MPI;
			else if (!strcasecmp(sub, "MPI_VERBOSE"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_MPI_VERBOSE;
			else if (!strcasecmp(sub, "HYP"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_HYP;
			else if (!strcasecmp(sub, "HYP_VERBOSE"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_HYP_VERBOSE;
			else if (!strcasecmp(sub, "TASK_VERBOSE_EXTRA"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_TASK_VERBOSE_EXTRA;
			else if (!strcasecmp(sub, "MPI_VERBOSE_EXTRA"))
				profiling_key_mask |= _STARPU_FUT_KEYMASK_MPI_VERBOSE_EXTRA;
			/* Added categories here should also be added in the documentation
			 * 501_environment_variable.doxy. */
			else
				_STARPU_MSG("Unknown event type '%s'\n", sub);
		}
	}
	else
	{
		/* If user doesn't want to filter events, all events are recorded: */
		profiling_key_mask = KEYMASKALL_DEFAULT;
	}

	return profiling_key_mask;
}

void starpu_profiling_set_id(int new_id)
{
	_STARPU_DEBUG("Set id to <%d>\n", new_id);
	_starpu_id = new_id;
	_starpu_profile_set_tracefile();

#ifdef HAVE_FUT_SET_FILENAME
	fut_set_filename(_starpu_prof_file_user);
#endif
}

void _starpu_profiling_set_mpi_worldsize(int worldsize)
{
	STARPU_ASSERT(worldsize >= 1);
	_starpu_mpi_worldsize = worldsize;

	int generate_trace = starpu_getenv_number("STARPU_GENERATE_TRACE");
	if (generate_trace == 1 && _starpu_mpi_worldsize > 1)
	{
		/** TODO: make it work !
		 * The problem is that when STARPU_GENERATE_TRACE is used, each MPI
		 * process will generate the trace corresponding to its own execution
		 * (which makes no sense in MPI execution with several processes).
		 * Although letting only one StarPU process generating the trace by
		 * using the trace files of all MPI processes is not the most
		 * complicated thing to do, one case is not easy to deal with: what to
		 * do when each process stored its trace file in the local memory of
		 * the node (e.g. /tmp/) ?
		 */
		_STARPU_MSG("You can't use STARPU_GENERATE_TRACE=1 with several MPI processes. Use starpu_fxt_tool after application execution.\n");
	}
}

void starpu_fxt_autostart_profiling(int autostart)
{
	/* By calling this function with autostart = 0 before starpu_init(),
	 * FxT will record only required event to properly work later (KEYMASK_META), and
	 * won't record anything else. */
	if (autostart)
		initial_key_mask = _starpu_profile_get_user_keymask();
	else
		initial_key_mask = _STARPU_FUT_KEYMASK_META;
}

void starpu_fxt_start_profiling()
{
	unsigned threadid = _starpu_gettid();
	fut_keychange(FUT_ENABLE, _starpu_profile_get_user_keymask(), threadid);
	_STARPU_TRACE_META("start_profiling");
}

void starpu_fxt_stop_profiling()
{
	unsigned threadid = _starpu_gettid();
	_STARPU_TRACE_META("stop_profiling");
	fut_keychange(FUT_SETMASK, _STARPU_FUT_KEYMASK_META, threadid);
}

int starpu_fxt_is_enabled()
{
	return starpu_getenv_number_default("STARPU_FXT_TRACE", 0);
}

#ifdef HAVE_FUT_SETUP_FLUSH_CALLBACK
void _starpu_fxt_flush_callback()
{
	_STARPU_MSG("FxT is flushing trace to disk ! This can impact performance.\n");
	_STARPU_MSG("Maybe you should increase the value of STARPU_TRACE_BUFFER_SIZE ?\n");

	starpu_fxt_trace_user_event_string("fxt flush");
}
#endif

void _starpu_fxt_init_profiling(uint64_t trace_buffer_size)
{
	unsigned threadid;

	STARPU_PTHREAD_MUTEX_LOCK(&_starpu_fxt_started_mutex);
	if (!(_starpu_fxt_willstart = starpu_fxt_is_enabled()))
	{
		STARPU_PTHREAD_COND_BROADCAST(&_starpu_fxt_started_cond);
		STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_fxt_started_mutex);
		return;
	}

	STARPU_ASSERT(!_starpu_fxt_started);

	_starpu_fxt_started = 1;
	_starpu_written = 0;
	_starpu_profile_set_tracefile();

	STARPU_HG_DISABLE_CHECKING(fut_active);

#ifdef HAVE_FUT_SET_FILENAME
	fut_set_filename(_starpu_prof_file_user);
#endif
#ifdef HAVE_ENABLE_FUT_FLUSH
	// when the event buffer is full, fxt stops recording events.
	// The trace may thus be incomplete.
	// Enable the fut_flush function which is called when the
	// fxt event buffer is full to flush the buffer to disk,
	// therefore allowing to record the remaining events.
	enable_fut_flush();
#endif

	threadid = _starpu_gettid();

#ifdef HAVE_FUT_SETUP_FLUSH_CALLBACK
	if (fut_setup_flush_callback(trace_buffer_size / sizeof(unsigned long), initial_key_mask, threadid, &_starpu_fxt_flush_callback) < 0)
#else
	if (fut_setup(trace_buffer_size / sizeof(unsigned long), initial_key_mask, threadid) < 0)
#endif
	{
		perror("fut_setup");
		STARPU_ABORT();
	}

	STARPU_PTHREAD_COND_BROADCAST(&_starpu_fxt_started_cond);
	STARPU_PTHREAD_MUTEX_UNLOCK(&_starpu_fxt_started_mutex);

	return;
}

int _starpu_generate_paje_trace_read_option(const char *option, struct starpu_fxt_options *options)
{
	if (strcmp(option, "-c") == 0)
	{
		options->per_task_colour = 1;
	}
	else if (strcmp(option, "-no-events") == 0)
	{
		options->no_events = 1;
	}
	else if (strcmp(option, "-no-counter") == 0)
	{
		options->no_counter = 1;
	}
	else if (strcmp(option, "-no-bus") == 0)
	{
		options->no_bus = 1;
	}
	else if (strcmp(option, "-no-flops") == 0)
	{
		options->no_flops = 1;
	}
	else if (strcmp(option, "-no-smooth") == 0)
	{
		options->no_smooth = 1;
	}
	else if (strcmp(option, "-no-acquire") == 0)
	{
		options->no_acquire = 1;
	}
	else if (strcmp(option, "-memory-states") == 0)
	{
		options->memory_states = 1;
	}
	else if (strcmp(option, "-internal") == 0)
	{
		options->internal = 1;
	}
	else if (strcmp(option, "-label-deps") == 0)
	{
		options->label_deps = 1;
	}
	else if (strcmp(option, "-number-events") == 0)
	{
		options->number_events_path = strdup("number_events.data");
	}
	else if (strcmp(option, "-use-task-color") == 0)
	{
		options->use_task_color = 1;
	}
	else
	{
		return 1;
	}
	return 0;
}

static void _starpu_generate_paje_trace(char *input_fxt_filename, char *output_paje_filename, char *dirname)
{
	/* We take default options */
	struct starpu_fxt_options options;
	starpu_fxt_options_init(&options);

	char *trace_options = starpu_getenv("STARPU_GENERATE_TRACE_OPTIONS");
	if (trace_options)
	{
		char *option = strtok(trace_options, " ");
		while (option)
		{
			int ret = _starpu_generate_paje_trace_read_option(option, &options);
			if (ret == 1)
				_STARPU_MSG("Option <%s> is not a valid option for starpu_fxt_tool\n", option);
			option = strtok(NULL, " ");
		}
	}

	options.ninputfiles = 1;
	options.filenames[0] = input_fxt_filename;
	free(options.out_paje_path);
	options.out_paje_path = strdup(output_paje_filename);
	options.file_prefix = "";
	options.file_rank = -1;
	options.dir = dirname;

	starpu_fxt_generate_trace(&options);
	starpu_fxt_options_shutdown(&options);
}

void _starpu_fxt_dump_file(void)
{
	if (!_starpu_fxt_started)
		return;

	char hostname[128];
	gethostname(hostname, 128);

	int ret = fut_endup(_starpu_prof_file_user);
	if (ret < 0)
		_STARPU_MSG("Problem when writing FxT traces into file %s:%s\n", hostname, _starpu_prof_file_user);
#ifdef STARPU_VERBOSE
	else
		_STARPU_MSG("Writing FxT traces into file %s:%s\n", hostname, _starpu_prof_file_user);
#endif
}

void _starpu_stop_fxt_profiling(void)
{
	if (!_starpu_fxt_started)
		return;
	if (!_starpu_written)
	{
		_starpu_fxt_dump_file();

		/* Should we generate a Paje trace directly ? */
		int generate_trace = starpu_getenv_number("STARPU_GENERATE_TRACE");
		if (_starpu_mpi_worldsize == 1 && generate_trace == 1)
		{
			_starpu_set_catch_signals(0);
			char *fxt_prefix = starpu_getenv("STARPU_FXT_PREFIX");
			_starpu_generate_paje_trace(_starpu_prof_file_user, "paje.trace", fxt_prefix);
		}

		int ret = fut_done();
		if (ret < 0)
		{
			/* Something went wrong with the FxT trace (eg. there
			 * was too many events) */
			_STARPU_MSG("Warning: the FxT trace could not be generated properly\n");
		}

		_starpu_written = 1;
		_starpu_fxt_started = 0;
	}
}

#else // STARPU_USE_FXT

void starpu_fxt_autostart_profiling(int autostart STARPU_ATTRIBUTE_UNUSED)
{
}

void starpu_fxt_start_profiling()
{
}

void starpu_fxt_stop_profiling()
{
}

#endif // STARPU_USE_FXT

void starpu_fxt_trace_user_event(unsigned long code STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	_STARPU_TRACE_USER_EVENT(code);
#endif
}

void starpu_fxt_trace_user_event_string(const char *s STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	_STARPU_TRACE_EVENT(s);
#endif
}
