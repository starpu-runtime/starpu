/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include <starpu.h>
#include <common/config.h>
#ifdef STARPU_USE_FXT

#include <common/fxt.h>

#define PROF_BUFFER_SIZE  (8*1024*1024)

static char PROF_FILE_USER[128];
static int fxt_started = 0;

static int written = 0;

static void profile_set_tracefile(char *fmt, ...)
{
	va_list vl;
	char *user;
	
	va_start(vl, fmt);
	vsprintf(PROF_FILE_USER, fmt, vl);
	va_end(vl);

	user = getenv("USER");
	if (!user)
		user = "";

	int pid = getpid();

	char suffix[128];
	snprintf(suffix, 128, "_user_%s_%d", user, pid);

	strcat(PROF_FILE_USER, suffix);
}

void _starpu_start_fxt_profiling(void)
{
	unsigned threadid;

	if (!fxt_started) {
		fxt_started = 1;

		char *fxt_prefix = getenv("STARPU_FXT_PREFIX");
		if (!fxt_prefix)
			fxt_prefix = "/tmp/prof_file";

		profile_set_tracefile(fxt_prefix);
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

#endif

void starpu_trace_user_event(unsigned code __attribute__((unused)))
{
#ifdef STARPU_USE_FXT
	STARPU_TRACE_USER_EVENT(code);
#endif
}
