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

/*
 * This is a small example of a C++ program using starpu.  We here just
 * add two std::vector without copying them (0 copy).
 */

#include <cassert>
#include <vector>

#ifdef PRINT_OUTPUT
#include <iostream>
#endif

#include <starpu.h>

void cpu_kernel_add_vectors(void *buffers[], void *cl_arg)
{
	// get the current task
	starpu_task* task = starpu_task_get_current();

	// get the user data (pointers to the vec_A, vec_B, vec_C std::vector)
	void* u_data0 = starpu_data_get_user_data(task->handles[0]); assert(u_data0);
	void* u_data1 = starpu_data_get_user_data(task->handles[1]); assert(u_data1);
	void* u_data2 = starpu_data_get_user_data(task->handles[2]); assert(u_data2);

	// cast void* in std::vector<char>*
	std::vector<char>* vec_A = static_cast<std::vector<char>*>(u_data0);
	std::vector<char>* vec_B = static_cast<std::vector<char>*>(u_data1);
	std::vector<char>* vec_C = static_cast<std::vector<char>*>(u_data2);

	// all the std::vector have to have the same size
	assert(vec_A->size() == vec_B->size() && vec_B->size() == vec_C->size());

	// performs the vector addition (vec_C[] = vec_A[] + vec_B[])
	for (size_t i = 0; i < vec_C->size(); i++)
		(*vec_C)[i] = (*vec_A)[i] + (*vec_B)[i];
}

#define VEC_SIZE 1024

int main(int argc, char **argv)
{
	std::vector<char> vec_A(VEC_SIZE, 2); // all the vector is initialized to 2
	std::vector<char> vec_B(VEC_SIZE, 3); // all the vector is initialized to 3
	std::vector<char> vec_C(VEC_SIZE, 0); // all the vector is initialized to 0

	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.nmic = 0;
	conf.nmpi_ms = 0;

	// initialize StarPU with default configuration
	int ret = starpu_init(&conf);
	if (ret == -ENODEV)
		return 77;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	/* StarPU can overwrite object if NUMA transfers are made */
	if (starpu_memory_nodes_get_numa_count() > 1)
	{
		starpu_shutdown();
		return 77;
	}

	// StarPU data registering
	starpu_data_handle_t spu_vec_A;
	starpu_data_handle_t spu_vec_B;
	starpu_data_handle_t spu_vec_C;

	// give the data of the vector to StarPU (C array)
	starpu_vector_data_register(&spu_vec_A, STARPU_MAIN_RAM, (uintptr_t)&vec_A[0], vec_A.size(), sizeof(char));
	starpu_vector_data_register(&spu_vec_B, STARPU_MAIN_RAM, (uintptr_t)&vec_B[0], vec_B.size(), sizeof(char));
	starpu_vector_data_register(&spu_vec_C, STARPU_MAIN_RAM, (uintptr_t)&vec_C[0], vec_C.size(), sizeof(char));

	// pass the pointer to the C++ vector object to StarPU
	starpu_data_set_user_data(spu_vec_A, (void*)&vec_A);
	starpu_data_set_user_data(spu_vec_B, (void*)&vec_B);
	starpu_data_set_user_data(spu_vec_C, (void*)&vec_C);

	// create the StarPU codelet
	starpu_codelet cl;
	starpu_codelet_init(&cl);
	cl.cpu_funcs     [0] = cpu_kernel_add_vectors;
	cl.cpu_funcs_name[0] = "cpu_kernel_add_vectors";
	cl.nbuffers          = 3;
	cl.modes         [0] = STARPU_R;
	cl.modes         [1] = STARPU_R;
	cl.modes         [2] = STARPU_W;
	cl.name              = "add_vectors";

	// submit a new StarPU task to execute
	ret = starpu_task_insert(&cl,
	                         STARPU_R, spu_vec_A,
	                         STARPU_R, spu_vec_B,
	                         STARPU_W, spu_vec_C,
	                         0);
	if (ret == -ENODEV)
	{
		// StarPU data unregistering
		starpu_data_unregister(spu_vec_C);
		starpu_data_unregister(spu_vec_B);
		starpu_data_unregister(spu_vec_A);

		// terminate StarPU, no task can be submitted after
		starpu_shutdown();

		return 77;
	}

	STARPU_CHECK_RETURN_VALUE(ret, "task_submit::add_vectors");

	// wait the task
	starpu_task_wait_for_all();

	// StarPU data unregistering
	starpu_data_unregister(spu_vec_C);
	starpu_data_unregister(spu_vec_B);
	starpu_data_unregister(spu_vec_A);

	// terminate StarPU, no task can be submitted after
	starpu_shutdown();

	// check results
	bool fail = false;
	int i = 0;
	while (!fail && i < VEC_SIZE)
		fail = vec_C[i++] != 5;

	if (fail)
	{
#ifdef PRINT_OUTPUT
		std::cout << "Example failed..." << std::endl;
#endif
		return EXIT_FAILURE;
	}
	else
	{
#ifdef PRINT_OUTPUT
		std::cout << "Example successfully passed!" << std::endl;
#endif
		return EXIT_SUCCESS;
	}
}
