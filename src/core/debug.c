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

#include <common/config.h>
#include <core/debug.h>
#include <common/utils.h>

#ifdef STARPU_VERBOSE
/* we want a single writer at the same time to have a log that is readable */
static starpu_pthread_mutex_t logfile_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static FILE *logfile = NULL;
#endif

int _starpu_debug
#ifdef STARPU_DEBUG
	= 1
#else
	= 0
#endif
	;

/* Tell gdb whether FXT is compiled in or not */
int _starpu_use_fxt
#ifdef STARPU_USE_FXT
	= 1
#endif
	;

void _starpu_open_debug_logfile(void)
{
#ifdef STARPU_VERBOSE
	/* what is  the name of the file ? default = "starpu.log" */
	char *logfile_name;

	logfile_name = starpu_getenv("STARPU_LOGFILENAME");
	if (!logfile_name)
	{
		logfile_name = "starpu.log";
	}

	logfile = fopen(logfile_name, "w+");
	STARPU_ASSERT_MSG(logfile, "Could not open file %s for verbose logs (%s). You can specify another file destination with the STARPU_LOGFILENAME environment variable", logfile_name, strerror(errno));
#endif
}

void _starpu_close_debug_logfile(void)
{
#ifdef STARPU_VERBOSE
	if (logfile)
	{
		fclose(logfile);
		logfile = NULL;
	}
#endif
}

void _starpu_print_to_logfile(const char *format STARPU_ATTRIBUTE_UNUSED, ...)
{
#ifdef STARPU_VERBOSE
	va_list args;
	va_start(args, format);
	STARPU_PTHREAD_MUTEX_LOCK(&logfile_mutex);
	vfprintf(logfile, format, args);
	STARPU_PTHREAD_MUTEX_UNLOCK(&logfile_mutex);
	va_end( args );
#endif
}

/* Record codelet to give ayudame nice function ids starting from 0. */
#if defined(STARPU_USE_AYUDAME1)
struct ayudame_codelet
{
	char *name;
	struct starpu_codelet *cl;
} *codelets;
static unsigned ncodelets, ncodelets_alloc;
static starpu_pthread_mutex_t ayudame_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
int64_t _starpu_ayudame_get_func_id(struct starpu_codelet *cl)
{
	unsigned i;
	const char *name;
	if (!cl)
		return 0;
	name = _starpu_codelet_get_model_name(cl);
	STARPU_PTHREAD_MUTEX_LOCK(&ayudame_mutex);
	for (i=0; i < ncodelets; i++)
	{
		if (codelets[i].cl == cl &&
			((!name && !codelets[i].name) ||
				((name && codelets[i].name) && !strcmp(codelets[i].name, name))))
		{
			STARPU_PTHREAD_MUTEX_UNLOCK(&ayudame_mutex);
			return i + 1;
		}
	}
	if (ncodelets == ncodelets_alloc)
	{
		if (!ncodelets_alloc)
			ncodelets_alloc = 16;
		else
			ncodelets_alloc *= 2;
		_STARPU_REALLOC(codelets, ncodelets_alloc * sizeof(*codelets));
	}
	codelets[ncodelets].cl = cl;
	if (name)
		/* codelet might be freed by user */
		codelets[ncodelets].name = strdup(name);
	else
		codelets[ncodelets].name = NULL;
	i = ncodelets++;
	if (name)
		AYU_event(AYU_REGISTERFUNCTION, i+1, (void*) name);
	STARPU_PTHREAD_MUTEX_UNLOCK(&ayudame_mutex);
	return i + 1;
}
#endif /* AYUDAME1 */
