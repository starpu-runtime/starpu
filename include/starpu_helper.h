/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2008-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __STARPU_HELPER_H__
#define __STARPU_HELPER_H__

#include <stdio.h>
#include <starpu.h>

#ifdef STARPU_HAVE_HWLOC
#include <hwloc.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

/**
   @defgroup API_Miscellaneous_Helpers Miscellaneous Helpers
   @{
*/

/**
   Return the min of the two parameters.
*/
#define STARPU_MIN(a,b)	((a)<(b)?(a):(b))
/**
   Return the max of the two parameters.
*/
#define STARPU_MAX(a,b)	((a)<(b)?(b):(a))

/**
   Define a value which can be used to mark pointers as invalid
   values.
*/
#define STARPU_POISON_PTR	((void *)0xdeadbeef)

extern int _starpu_silent;

char *starpu_getenv(const char *str);

/**
   If the environment variable \c str is defined and its value is contained in the array \c strings, return the array position.
   Raise an error if the environment variable \c str is defined with a value not in \c strings
   Return \c defvalue if the environment variable \c str is not defined.
 */
int starpu_get_env_string_var_default(const char *str, const char *strings[], int defvalue);

/**
   If the environment variable \c str is defined with a well-defined size value, return the value as a size in bytes. Expected size qualifiers are b, B, k, K, m, M, g, G. The default qualifier is K.
   If the environment variable \c str is not defined or is empty, return \c defval
   Raise an error if the value of the environment variable \c str is not well-defined.
 */
int starpu_get_env_size_default(const char *str, int defval);

/**
   Return the integer value of the environment variable named \p str.
   Return 0 otherwise (the variable does not exist or has a
   non-integer value).
*/
static __starpu_inline int starpu_get_env_number(const char *str)
{
	char *strval;

	strval = starpu_getenv(str);
	if (strval)
	{
		/* the env variable was actually set */
		long int val;
		char *pcheck;

		val = strtol(strval, &pcheck, 10);
		if (*pcheck)
		{
			fprintf(stderr,"The %s environment variable must contain an integer\n", str);
			STARPU_ABORT();
		}

		/* fprintf(stderr, "ENV %s WAS %d\n", str, val); */
		STARPU_ASSERT_MSG(val >= 0, "The value for the environment variable '%s' cannot be negative", str);
		return (int)val;
	}
	else
	{
		/* there is no such env variable */
		/* fprintf("There was no %s ENV\n", str); */
		return -1;
	}
}

static __starpu_inline int starpu_get_env_number_default(const char *str, int defval)
{
	int ret = starpu_get_env_number(str);
	if (ret == -1)
		ret = defval;
	return ret;
}

static __starpu_inline float starpu_get_env_float_default(const char *str, float defval)
{
	char *strval;

	strval = starpu_getenv(str);
	if (strval)
	{
		/* the env variable was actually set */
		float val;
		char *pcheck;

		val = strtof(strval, &pcheck);
		if (*pcheck)
		{
			fprintf(stderr,"The %s environment variable must contain a float\n", str);
			STARPU_ABORT();
		}

		/* fprintf(stderr, "ENV %s WAS %f\n", str, val); */
		return val;
	}
	else
	{
		/* there is no such env variable */
		/* fprintf("There was no %s ENV\n", str); */
		return defval;
	}
}

/**
   Execute the given function \p func on a subset of workers. When
   calling this method, the offloaded function \p func is executed by
   every StarPU worker that are eligible to execute the function. The
   argument \p arg is passed to the offloaded function. The argument
   \p where specifies on which types of processing units the function
   should be executed.
   Similarly to the field starpu_codelet::where, it is possible to
   specify that the function should be executed on every CUDA device
   and every CPU by passing ::STARPU_CPU|::STARPU_CUDA. This function
   blocks until \p func has been executed on every appropriate
   processing units, and thus may not be called from a callback
   function for instance.
*/
void starpu_execute_on_each_worker(void (*func)(void *), void *arg, uint32_t where);

/**
   Same as starpu_execute_on_each_worker(), except that the task name
   is specified in the argument \p name.
*/
void starpu_execute_on_each_worker_ex(void (*func)(void *), void *arg, uint32_t where, const char *name);

/**
   Call \p func(\p arg) on every worker in the \p workers array. \p
   num_workers indicates the number of workers in this array.  This
   function is synchronous, but the different workers may execute the
   function in parallel.
*/
void starpu_execute_on_specific_workers(void (*func)(void*), void *arg, unsigned num_workers, unsigned *workers, const char *name);

/**
   Return the current date in micro-seconds.
*/
double starpu_timing_now(void);

/**
   Copy the content of \p src_handle into \p dst_handle. The parameter \p
   asynchronous indicates whether the function should block or not. In
   the case of an asynchronous call, it is possible to synchronize with
   the termination of this operation either by the means of implicit
   dependencies (if enabled) or by calling starpu_task_wait_for_all(). If
   \p callback_func is not <c>NULL</c>, this callback function is executed after
   the handle has been copied, and it is given the pointer \p
   callback_arg as argument.
*/
int starpu_data_cpy(starpu_data_handle_t dst_handle, starpu_data_handle_t src_handle, int asynchronous, void (*callback_func)(void*), void *callback_arg);

/**
   Call hwloc-ps to display binding of each processus and thread running on
   the machine.<br>
   Use the environment variable \ref STARPU_DISPLAY_BINDINGS to automatically
   call this function at the beginning of the execution of StarPU.
*/
void starpu_display_bindings(void);

/**
   If \c hwloc is used, convert the given \p logical_index of a PU to the OS
   index of this PU. If \c hwloc is not used, return \p logical_index.
*/
int starpu_get_pu_os_index(unsigned logical_index);

#ifdef STARPU_HAVE_HWLOC
/**
   Get the hwloc topology used by StarPU. One can use this pointer to get
   information about topology, but not to change settings related to topology.
*/
hwloc_topology_t starpu_get_hwloc_topology(void);
#endif
/** @} */

#ifdef __cplusplus
}
#endif

#endif // __STARPU_HELPER_H__
