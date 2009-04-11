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

#include <common/fxt.h>

#define PROF_BUFFER_SIZE  (8*1024*1024)

static char PROF_FILE_USER[128];
static int fxt_started = 0;

static void profile_stop(void)
{
	fut_endup(PROF_FILE_USER);
}

static void profile_set_tracefile(char *fmt, ...)
{
	va_list vl;
	
	va_start(vl, fmt);
	vsprintf(PROF_FILE_USER, fmt, vl);
	va_end(vl);
	strcat(PROF_FILE_USER, "_user_");
}


void start_fxt_profiling(void)
{
	unsigned threadid;

	if (!fxt_started) {
		fxt_started = 1;
		profile_set_tracefile("/tmp/prof_file");
	}

	threadid = syscall(SYS_gettid);

	atexit(profile_stop);

	if(fut_setup(PROF_BUFFER_SIZE, FUT_KEYMASKALL, threadid) < 0) {
		perror("fut_setup");
		STARPU_ASSERT(0);
	}

	fut_keychange(FUT_ENABLE, FUT_KEYMASKALL, threadid);

	return;
}

void fxt_register_thread(unsigned coreid)
{
	FUT_DO_PROBE2(FUT_NEW_LWP_CODE, coreid, syscall(SYS_gettid));
}
