/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2018                                     Alexis Juven
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
 * jlstarpu_task_submit.c
 *
 *  Created on: 27 juin 2018
 *      Author: ajuven
 */


#include "jlstarpu.h"


struct starpu_codelet * jlstarpu_new_codelet()
{
	struct starpu_codelet * output;
	TYPE_MALLOC(output, 1);

	starpu_codelet_init(output);

	return output;
}


#if 0
struct starpu_codelet * jlstarpu_translate_codelet(struct jlstarpu_codelet * const input)
{
	struct starpu_codelet * output;
	TYPE_MALLOC(output, 1);

	starpu_codelet_init(output);



	output->where = input->where;

	output->cpu_funcs[0] = input->cpu_func;
	output->cpu_funcs_name[0] = input->cpu_func_name;

	output->cuda_funcs[0] = input->cuda_func;

	output->nbuffers = input->nbuffer;
	memcpy(&(output->modes), input->modes, input->nbuffer * sizeof(enum starpu_data_access_mode));

	output->model = input->model;

	return output;
}
#endif

void jlstarpu_codelet_update(const struct jlstarpu_codelet * const input, struct starpu_codelet * const output)
{
	output->where = input->where;

	output->cpu_funcs[0] = input->cpu_func;
	output->cpu_funcs_name[0] = input->cpu_func_name;

	output->cuda_funcs[0] = input->cuda_func;

	output->nbuffers = input->nbuffer;
	memcpy(&(output->modes), input->modes, input->nbuffer * sizeof(enum starpu_data_access_mode));

	output->model = input->model;

}
#if 0
void jlstarpu_free_codelet(struct starpu_codelet * cl)
{
	free(cl);
}
#endif



#if 0
struct starpu_task * jlstarpu_translate_task(const struct jlstarpu_task * const input)
{
	struct starpu_task * output = starpu_task_create();

	if (output == NULL){
		return NULL;
	}

	output->cl = input->cl;
	memcpy(&(output->handles), input->handles, input->cl->nbuffers * sizeof(starpu_data_handle_t));
	output->synchronous = input->synchronous;


	return output;
}
#endif



void jlstarpu_task_update(const struct jlstarpu_task * const input, struct starpu_task * const output)
{
	output->cl = input->cl;
	memcpy(&(output->handles), input->handles, input->cl->nbuffers * sizeof(starpu_data_handle_t));
	output->synchronous = input->synchronous;
	output->cl_arg = input->cl_arg;
	output->cl_arg_size = input->cl_arg_size;
}



/*

void print_perfmodel(struct starpu_perfmodel * p)
{
	printf("Perfmodel at address %p:\n");
	printf("\ttype : %u\n", p->type);
	printf("\tcost_function : %p\n", p->cost_function);
	printf("\tarch_cost_function : %p\n", p->arch_cost_function);
	printf("\tsize_base : %p\n", p->size_base);
	printf("\tfootprint : %p\n", p->footprint);
	printf("\tsymbol : %s\n", p->symbol);
	printf("\tis_loaded : %u\n", p->is_loaded);
	printf("\tbenchmarking : %u\n", p->benchmarking);
	printf("\tis_init : %u\n", p->is_init);
	printf("\tparameters : %p\n", p->parameters);
	printf("\tparameters_names : %p\n", p->parameters_names);
	printf("\tnparameters : %u\n", p->nparameters);
	printf("\tcombinations : %p\n", p->combinations);
	printf("\tncombinations : %u\n", p->ncombinations);
	printf("\tstate : %p\n", p->state);

}


*/

#if 0
/*
 * TODO : free memory
 */
int jlstarpu_task_submit(const struct jlstarpu_task * const jl_task)
{
	DEBUG_PRINT("Inside C wrapper");

	struct starpu_task * task;
	int ret_code;


	DEBUG_PRINT("Translating task...");
	task = jlstarpu_translate_task(jl_task);

	if (task == NULL){
		fprintf(stderr, "Error while creating the task.\n");
		return EXIT_FAILURE;
	}

	DEBUG_PRINT("Task translated");
	DEBUG_PRINT("Submitting task to StarPU...");
	ret_code = starpu_task_submit(task);
	DEBUG_PRINT("starpu_task_submit has returned");


	if (ret_code != 0){
		fprintf(stderr, "Error while submitting task.\n");
		return ret_code;
	}


	DEBUG_PRINT("Done");
	DEBUG_PRINT("END OF STARPU FUNCTION");


	return ret_code;
}

#endif







#define JLSTARPU_UPDATE_FUNC(type, field)\
	\
	void jlstarpu_##type##_update_##field(const struct jlstarpu_##type * const input, struct starpu_##type * const output)\
	{\
		output->field = input->field;\
	}










