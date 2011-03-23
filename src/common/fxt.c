/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009, 2010-2011  Université de Bordeaux 1
 * Copyright (C) 2010, 2011  Centre National de la Recherche Scientifique
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

#ifdef STARPU_HAVE_WINDOWS
#include <windows.h>
#endif
		
#define PROF_BUFFER_SIZE  (8*1024*1024)

static char PROF_FILE_USER[128];
static int fxt_started = 0;

static int written = 0;

static int id;

static void _profile_set_tracefile(void *last, ...)
{
	va_list vl;
	char *user;

        char *fxt_prefix = getenv("STARPU_FXT_PREFIX");
        if (!fxt_prefix)
			fxt_prefix = "/tmp/";

	va_start(vl, last);
	vsprintf(PROF_FILE_USER, fxt_prefix, vl);
	va_end(vl);

	user = getenv("USER");
	if (!user)
		user = "";

	char suffix[128];
	snprintf(suffix, 128, "prof_file_%s_%d", user, id);

	strcat(PROF_FILE_USER, suffix);
}

void starpu_set_profiling_id(int new_id) {
        _STARPU_DEBUG("Set id to <%d>\n", new_id);
	id = new_id;
        _profile_set_tracefile(NULL);
}

void _starpu_start_fxt_profiling(void)
{
	unsigned threadid;

	if (!fxt_started) {
		fxt_started = 1;
		_profile_set_tracefile(NULL);
	}

	threadid = syscall(SYS_gettid);

	atexit(_starpu_stop_fxt_profiling);

	if(fut_setup(PROF_BUFFER_SIZE, FUT_KEYMASKALL, threadid) < 0) {
		perror("fut_setup");
		STARPU_ABORT();
	}

	fut_keychange(FUT_ENABLE, FUT_KEYMASKALL, threadid);

	return;
}

static void generate_paje_trace(char *input_fxt_filename, char *output_paje_filename)
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
	if (!written)
	{
#ifdef STARPU_VERBOSE
	        char hostname[128];
		gethostname(hostname, 128);
		fprintf(stderr, "Writing FxT traces into file %s:%s\n", hostname, PROF_FILE_USER);
#endif
		fut_endup(PROF_FILE_USER);

		/* Should we generate a Paje trace directly ? */
		int generate_trace = starpu_get_env_number("STARPU_GENERATE_TRACE");
		if (generate_trace == 1)
			generate_paje_trace(PROF_FILE_USER, "paje.trace");

		int ret = fut_done();
		if (ret < 0)
		{
			/* Something went wrong with the FxT trace (eg. there
			 * was too many events) */
			fprintf(stderr, "Warning: the FxT trace could not be generated properly\n");
		}

		written = 1;
	}
}

void _starpu_fxt_register_thread(unsigned cpuid)
{
	FUT_DO_PROBE2(FUT_NEW_LWP_CODE, cpuid, syscall(SYS_gettid));
}

#endif // STARPU_USE_FXT

void starpu_trace_user_event(unsigned long code __attribute__((unused)))
{
#ifdef STARPU_USE_FXT
	STARPU_TRACE_USER_EVENT(code);
#endif
}
