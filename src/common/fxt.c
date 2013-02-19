/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2013  Université de Bordeaux 1
 * Copyright (C) 2010, 2011, 2012  Centre National de la Recherche Scientifique
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
#include <starpu_util.h>
#include <starpu_profiling.h>

#ifdef STARPU_USE_FXT
#include <common/fxt.h>
#include <starpu_fxt.h>

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif

#ifdef __linux__
#include <sys/syscall.h>   /* for SYS_gettid */
#elif defined(__FreeBSD__)
#include <sys/thr.h>       /* for thr_self() */
#endif

static char _STARPU_PROF_FILE_USER[128];
static int _starpu_fxt_started = 0;

static int _starpu_written = 0;

static int _starpu_id;

#ifdef STARPU_SIMGRID
/* Give virtual time to FxT */
uint64_t fut_getstamp(void)
{
	return starpu_timing_now()*1000.;
}
#endif

long _starpu_gettid(void)
{
#ifdef STARPU_SIMGRID
	return (uintptr_t) MSG_process_self();
#else
#if defined(__linux__)
	return syscall(SYS_gettid);
#elif defined(__FreeBSD__)
	long tid;
	thr_self(&tid);
	return tid;
#elif defined(__MINGW32__)
	return (long) GetCurrentThreadId();
#else
	return (long) pthread_self();
#endif
#endif
}

static void _starpu_profile_set_tracefile(void *last, ...)
{
	va_list vl;
	char *user;

	char *fxt_prefix = getenv("STARPU_FXT_PREFIX");
	if (!fxt_prefix)
	     fxt_prefix = "/tmp/";

	va_start(vl, last);
	vsprintf(_STARPU_PROF_FILE_USER, fxt_prefix, vl);
	va_end(vl);

	user = getenv("USER");
	if (!user)
		user = "";

	char suffix[128];
	snprintf(suffix, 128, "prof_file_%s_%d", user, _starpu_id);

	strcat(_STARPU_PROF_FILE_USER, suffix);
}

void starpu_set_profiling_id(int new_id)
{
	_STARPU_DEBUG("Set id to <%d>\n", new_id);
	_starpu_id = new_id;
	_starpu_profile_set_tracefile(NULL);

#ifdef HAVE_FUT_SET_FILENAME
	fut_set_filename(_STARPU_PROF_FILE_USER);
#endif
}

void starpu_fxt_start_profiling()
{
	unsigned threadid = _starpu_gettid();
	fut_keychange(FUT_ENABLE, FUT_KEYMASKALL, threadid);
}

void starpu_fxt_stop_profiling()
{
	unsigned threadid = _starpu_gettid();
	fut_keychange(FUT_DISABLE, FUT_KEYMASKALL, threadid);
}

void _starpu_init_fxt_profiling(unsigned trace_buffer_size)
{
	unsigned threadid;

	if (!_starpu_fxt_started)
	{
		_starpu_fxt_started = 1;
		_starpu_written = 0;
		_starpu_profile_set_tracefile(NULL);
	}

#ifdef HAVE_FUT_SET_FILENAME
	fut_set_filename(_STARPU_PROF_FILE_USER);
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

	atexit(_starpu_stop_fxt_profiling);

	unsigned int key_mask = FUT_KEYMASKALL;

	if (fut_setup(trace_buffer_size / sizeof(unsigned long), key_mask, threadid) < 0)
	{
		perror("fut_setup");
		STARPU_ABORT();
	}

	fut_keychange(FUT_ENABLE, key_mask, threadid);

	return;
}

static void _starpu_generate_paje_trace(char *input_fxt_filename, char *output_paje_filename)
{
	/* We take default options */
	struct starpu_fxt_options options;
	starpu_fxt_options_init(&options);

	/* TODO parse some STARPU_GENERATE_TRACE_OPTIONS env variable */

	options.ninputfiles = 1;
	options.filenames[0] = input_fxt_filename;
	options.out_paje_path = output_paje_filename;
	options.file_prefix = "";
	options.file_rank = -1;

	starpu_fxt_generate_trace(&options);
}

void _starpu_stop_fxt_profiling(void)
{
	if (!_starpu_written)
	{
#ifdef STARPU_VERBOSE
	        char hostname[128];
		gethostname(hostname, 128);
		fprintf(stderr, "Writing FxT traces into file %s:%s\n", hostname, _STARPU_PROF_FILE_USER);
#endif
		fut_endup(_STARPU_PROF_FILE_USER);

		/* Should we generate a Paje trace directly ? */
		int generate_trace = starpu_get_env_number("STARPU_GENERATE_TRACE");
		if (generate_trace == 1)
			_starpu_generate_paje_trace(_STARPU_PROF_FILE_USER, "paje.trace");

		int ret = fut_done();
		if (ret < 0)
		{
			/* Something went wrong with the FxT trace (eg. there
			 * was too many events) */
			fprintf(stderr, "Warning: the FxT trace could not be generated properly\n");
		}

		_starpu_written = 1;
		_starpu_fxt_started = 0;
	}
}

void _starpu_fxt_register_thread(unsigned cpuid)
{
	FUT_DO_PROBE2(FUT_NEW_LWP_CODE, cpuid, _starpu_gettid());
}

#endif // STARPU_USE_FXT
void starpu_fxt_start_profiling()
{
}

void starpu_fxt_stop_profiling()
{
}

#endif // STARPU_USE_FXT

void starpu_trace_user_event(unsigned long code STARPU_ATTRIBUTE_UNUSED)
{
#ifdef STARPU_USE_FXT
	_STARPU_TRACE_USER_EVENT(code);
#endif
}
