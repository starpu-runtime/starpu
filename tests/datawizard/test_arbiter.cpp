/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
 * for i from 0 to nbA
 *  insert task handles[i] in STARPU_RW|STARPU_COMMUTE
 *  for j from 0 to nbA
 *	  if i != j
			 insert task handles[i] in STARPU_RW|STARPU_COMMUTE, and handles[j] in STARPU_RW|STARPU_COMMUTE
 */


// @FUSE_STARPU


#include <starpu.h>
#include "../helper.h"

#include <vector>
#include <unistd.h>

#ifdef STARPU_QUICK_CHECK
#define SLEEP_SLOW 6000
#define SLEEP_FAST 1000
#elif !defined(STARPU_LONG_CHECK)
#define SLEEP_SLOW 60000
#define SLEEP_FAST 10000
#else
#define SLEEP_SLOW 600000
#define SLEEP_FAST 100000
#endif

static unsigned nb, nb_slow;

void callback(void * /*buffers*/[], void * /*cl_arg*/)
{
	unsigned val;
	val = STARPU_ATOMIC_ADD(&nb, 1);
	FPRINTF(stdout,"callback in (%u)\n", val); fflush(stdout);
	usleep(SLEEP_FAST);
	val = STARPU_ATOMIC_ADD(&nb, -1);
	FPRINTF(stdout,"callback out (%u)\n", val); fflush(stdout);
}

void callback_slow(void * /*buffers*/[], void * /*cl_arg*/)
{
	unsigned val;
	val = STARPU_ATOMIC_ADD(&nb_slow, 1);
	FPRINTF(stdout,"callback_slow in (%u)\n", val); fflush(stdout);
	usleep(SLEEP_SLOW);
	val = STARPU_ATOMIC_ADD(&nb_slow, -1);
	FPRINTF(stdout,"callback_slow out (%u)\n", val); fflush(stdout);
}


int main(int /*argc*/, char** /*argv*/)
{
	int ret;
	struct starpu_conf conf;
	starpu_arbiter_t arbiter, arbiter2;
	ret = starpu_conf_init(&conf);
	STARPU_ASSERT(ret == 0);
	//conf.ncpus = 1;//// 4
	ret = starpu_init(&conf);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_ASSERT(ret == 0);

	FPRINTF(stdout, "Max Thread %u\n", starpu_worker_get_count());

	//////////////////////////////////////////////////////

	starpu_codelet normalCodelete;
	{
		memset(&normalCodelete, 0, sizeof(normalCodelete));
		normalCodelete.where = STARPU_CPU;
		normalCodelete.cpu_funcs[0] = callback;
		normalCodelete.nbuffers = 2;
		normalCodelete.modes[0] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
		normalCodelete.modes[1] = starpu_data_access_mode(STARPU_RW|STARPU_COMMUTE);
		normalCodelete.name = "normalCodelete";
	}
	starpu_codelet slowCodelete;
	{
		memset(&slowCodelete, 0, sizeof(slowCodelete));
		slowCodelete.where = STARPU_CPU;
		slowCodelete.cpu_funcs[0] = callback_slow;
		slowCodelete.nbuffers = 1;
		slowCodelete.modes[0] = starpu_data_access_mode (STARPU_RW|STARPU_COMMUTE);
		slowCodelete.name = "slowCodelete";
	}

	//////////////////////////////////////////////////////
	//////////////////////////////////////////////////////

	///const int nbA = 3;
	const int nbA = 10;
	FPRINTF(stdout, "Nb A = %d\n", nbA);

	std::vector<starpu_data_handle_t> handleA(nbA);
	std::vector<int> dataA(nbA);
	arbiter = starpu_arbiter_create();
	arbiter2 = starpu_arbiter_create();
	for(int idx = 0 ; idx < nbA ; ++idx)
	{
		dataA[idx] = idx;
	}
	for(int idxHandle = 0 ; idxHandle < nbA ; ++idxHandle)
	{
		starpu_variable_data_register(&handleA[idxHandle], 0, (uintptr_t)&dataA[idxHandle], sizeof(dataA[idxHandle]));
		starpu_data_assign_arbiter(handleA[idxHandle], arbiter);
	}

	//////////////////////////////////////////////////////
	//////////////////////////////////////////////////////
	FPRINTF(stdout,"Submit tasks\n");

	for(int idxHandleA1 = 0 ; idxHandleA1 < nbA ; ++idxHandleA1)
	{
		ret = starpu_task_insert(&slowCodelete,
				(STARPU_RW|STARPU_COMMUTE), handleA[idxHandleA1],
				0);
		if (ret == -ENODEV) goto out;
		for(int idxHandleA2 = 0 ; idxHandleA2 < nbA ; ++idxHandleA2)
		{
			if(idxHandleA1 != idxHandleA2)
			{
				ret = starpu_task_insert(&normalCodelete,
						(STARPU_RW|STARPU_COMMUTE), handleA[idxHandleA1],
						(STARPU_RW|STARPU_COMMUTE), handleA[idxHandleA2],
						0);
				if (ret == -ENODEV) goto out;
			}
		}
	}

	//////////////////////////////////////////////////////
	FPRINTF(stdout,"Wait task\n");

	starpu_task_wait_for_all();

	//////////////////////////////////////////////////////
	FPRINTF(stdout,"Release data\n");

	for(int idxHandle = 0 ; idxHandle < nbA ; ++idxHandle)
	{
		starpu_data_unregister(handleA[idxHandle]);
	}

	//////////////////////////////////////////////////////
	FPRINTF(stdout,"Proceed gain, with several arbiters\n");

	for(int idxHandle = 0 ; idxHandle < nbA ; ++idxHandle)
	{
		starpu_variable_data_register(&handleA[idxHandle], 0, (uintptr_t)&dataA[idxHandle], sizeof(dataA[idxHandle]));
		starpu_data_assign_arbiter(handleA[idxHandle], (idxHandle%2)?arbiter:arbiter2);
	}

	//////////////////////////////////////////////////////
	//////////////////////////////////////////////////////
	FPRINTF(stdout,"Submit tasks\n");

	for(int idxHandleA1 = 0 ; idxHandleA1 < nbA ; ++idxHandleA1)
	{
		ret = starpu_task_insert(&slowCodelete,
				(STARPU_RW|STARPU_COMMUTE), handleA[idxHandleA1],
				0);
		if (ret == -ENODEV) goto out;
		for(int idxHandleA2 = 0 ; idxHandleA2 < nbA ; ++idxHandleA2)
		{
			if(idxHandleA1 != idxHandleA2)
			{
				ret = starpu_task_insert(&normalCodelete,
						(STARPU_RW|STARPU_COMMUTE), handleA[idxHandleA1],
						(STARPU_RW|STARPU_COMMUTE), handleA[idxHandleA2],
						0);
				if (ret == -ENODEV) goto out;
			}
		}
	}

	//////////////////////////////////////////////////////
	FPRINTF(stdout,"Wait task\n");

out:
	starpu_task_wait_for_all();

	//////////////////////////////////////////////////////
	FPRINTF(stdout,"Release data\n");

	for(int idxHandle = 0 ; idxHandle < nbA ; ++idxHandle)
	{
		starpu_data_unregister(handleA[idxHandle]);
	}
	starpu_arbiter_destroy(arbiter);
	starpu_arbiter_destroy(arbiter2);

	//////////////////////////////////////////////////////
	FPRINTF(stdout,"Shutdown\n");

	starpu_shutdown();

	return 0;
}
