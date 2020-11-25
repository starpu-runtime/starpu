/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
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
#include "../helper.h"
#include <stdlib.h>

/*
 * Try initializing starpu with various CPU parameters
 */

#if !defined(STARPU_HAVE_UNSETENV) || !defined(STARPU_HAVE_SETENV) || !defined(STARPU_USE_CPU)
#warning unsetenv or setenv are not defined. Or CPU are not enabled. Skipping test
int main(void)
{
	return STARPU_TEST_SKIPPED;
}
#else

static int check_cpu(int env_cpu, int conf_cpu, int precedence_over_env, int expected_cpu, int *cpu)
{
	int ret;

	FPRINTF(stderr, "\nTesting with env=%d - conf=%d - expected %d (ignore env %d)\n", env_cpu, conf_cpu, expected_cpu, precedence_over_env);

	if (env_cpu != -1)
	{
		char string[11];
		snprintf(string, sizeof(string), "%d", env_cpu);
		setenv("STARPU_NCPUS", string, 1);
	}

	struct starpu_conf user_conf;
	starpu_conf_init(&user_conf);
	user_conf.precedence_over_environment_variables = precedence_over_env;

	if (conf_cpu != -1)
	{
		user_conf.ncpus = conf_cpu;
	}
	ret = starpu_init(&user_conf);

	if (env_cpu != -1)
	{
		unsetenv("STARPU_NCPUS");
	}

	if (ret == -ENODEV)
	{
		return STARPU_TEST_SKIPPED;
	}
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	*cpu = starpu_cpu_worker_get_count();
	starpu_shutdown();

	if (expected_cpu == -1)
	{
		FPRINTF(stderr, "Number of CPUS: %3d\n", *cpu);
		return 0;
	}
	else
	{
		FPRINTF(stderr, "Number of CPUS: %3d -- Number of expected CPUs: %3d    --> %s\n", *cpu, expected_cpu, *cpu==expected_cpu?"SUCCESS":"FAILURE");
		return *cpu != expected_cpu;
	}
}

int main(void)
{
	int ret;
	int cpu, cpu_init;
	int cpu_test1, cpu_test2, cpu_test3;

	unsetenv("STARPU_NCPUS");
	unsetenv("STARPU_NCPU");

	ret = check_cpu(-1, -1, 0, -1, &cpu_init);
	if (ret) return ret;
	if (cpu_init <= 1) return STARPU_TEST_SKIPPED;

	if (cpu_init >= STARPU_MAXCPUS-5)
	{
		cpu_test1 = cpu_init-1;
		cpu_test2 = cpu_init-2;
		cpu_test3 = cpu_init-3;
	}
	else
	{
		cpu_test1 = cpu_init+1;
		cpu_test2 = cpu_init+2;
		cpu_test3 = cpu_init+3;
	}

	ret = check_cpu(cpu_test1, -1, 0, cpu_test1, &cpu);
	if (ret) return ret;

	// Do not set anything --> default value
	ret = check_cpu(-1, -1, 0, -1, &cpu);
	if (ret) return ret;
	if (cpu != cpu_init)
	{
		FPRINTF(stderr, "The number of CPUs is incorrect\n");
		return 1;
	}

	// Do not set environment variable, set starpu_conf::ncpus --> starpu_conf::ncpus
	ret = check_cpu(-1, cpu_test2, 0, cpu_test2, &cpu);
	if (ret) return ret;

	// Set environment variable, and do not set starpu_conf::ncpus --> starpu_conf::ncpus
	ret = check_cpu(cpu_test2, -1, 0, cpu_test2, &cpu);
	if (ret) return ret;

	// Set both environment variable and starpu_conf::ncpus --> environment variable
	ret = check_cpu(cpu_test3, cpu_test1, 0, cpu_test3, &cpu);
	if (ret) return ret;

	// Set both environment variable and starpu_conf::ncpus AND prefer starpu_conf over env --> starpu_conf::ncpus
	ret = check_cpu(cpu_test3, cpu_test1, 1, cpu_test1, &cpu);
	if (ret) return ret;

	// Set environment variable, and do no set starpu_conf, AND prefer starpu_conf over env --> environment variable
	ret = check_cpu(cpu_test2, -1, 1, cpu_test2, &cpu);
	if (ret) return ret;

	return 0;
}

 #endif
