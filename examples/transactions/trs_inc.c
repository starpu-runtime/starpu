/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * This is just a small example which increments two values of a vector several times.
 */
#include <starpu.h>

#ifdef STARPU_QUICK_CHECK
static unsigned niter = 500;
#elif !defined(STARPU_LONG_CHECK)
static unsigned niter = 5000;
#else
static unsigned niter = 50000;
#endif

#define DO_TRANS_MOD 10
#define DO_START_MOD 2

#define FPRINTF(ofile, fmt, ...) do { if (!getenv("STARPU_SSILENT")) {fprintf(ofile, fmt, ## __VA_ARGS__); }} while(0)

static int _do_start_transaction(int val)
{
	if ((val / DO_TRANS_MOD) % DO_START_MOD == 0)
	{
		return 0;
	}
	else
	{
		return 1;
	}
}

int do_start_transaction(void *descr, void *arg)
{
	int val = (int)(intptr_t)arg;
	int ret = _do_start_transaction(val);
	return ret;
}

void cpu_func(void *descr[], void *_args)
{
	(void)_args;
	int *val = (int *)STARPU_VARIABLE_GET_PTR(descr[0]);
	*val += 1;
}

int main(int argc, char **argv)
{
	int ret = 0;
	double start;
	double end;
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.nmpi_ms = 0;
	conf.ntcpip_ms = 0;

	ret = starpu_init(&conf);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	if (argc == 2)
		niter = atoi(argv[1]);

	int value = 0;

	starpu_data_handle_t handle;
	starpu_variable_data_register(&handle, STARPU_MAIN_RAM, (uintptr_t)&value, sizeof(value));

	struct starpu_codelet cl =
	{
		.cpu_funcs	= { cpu_func },
		.cpu_funcs_name	= { "cpu_func" },
		.nbuffers	= STARPU_VARIABLE_NBUFFERS,
		.name	= "trs_increment"
	};

	start = starpu_timing_now();

	struct starpu_transaction *transaction = starpu_transaction_open(do_start_transaction, (void*)(intptr_t)0);
	if (transaction == NULL)
	{
		starpu_cublas_shutdown();
		starpu_shutdown();
		return 77; /* transaction begin task submit failed with ENODEV */
	}

	int simulated_transaction_status = _do_start_transaction(0);
	int expected_result = 0;
	unsigned i;

	for (i = 0; i < niter; i++)
	{
		if (i>0 && (i%DO_TRANS_MOD == 0))
		{
			starpu_transaction_next_epoch(transaction, (void*)(intptr_t)i);
			simulated_transaction_status = _do_start_transaction(i);
		}

		if (simulated_transaction_status)
		{
			expected_result ++;
		}

		ret = starpu_task_insert(&cl,
					 STARPU_RW, handle,
					 STARPU_TRANSACTION, transaction,
					 0);

		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			FPRINTF(stderr, "No worker may execute this task\n");
			starpu_data_unregister(handle);
			goto enodev;
		}
		STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
	}

	starpu_transaction_close(transaction);

	starpu_task_wait_for_all();

	starpu_data_unregister(handle);

	end = starpu_timing_now();

	if (value != expected_result)
	{
		FPRINTF(stderr, "Incorrect result, value = %d, expected %d\n", value, expected_result);
		ret = 1;
	}

	double timing = end - start;

	FPRINTF(stderr, "%u,%f,%d\n", niter, timing/1000, value);

enodev:
	starpu_shutdown();

	return (ret == -ENODEV ? 77 : ret);
}
