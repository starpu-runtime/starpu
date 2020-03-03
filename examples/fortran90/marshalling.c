/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2015       ONERA
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

/* Helper functions to initialize StarPU and register element matrices */

#include <starpu.h>

//--------------------------------------------------------------//
void starpu_register_element_c(int Neq_max,int Np, int Ng,double **ro, double **dro,
                               double **basis, void **ro_h, void **dro_h, void **basis_h)
{
	starpu_data_handle_t ro_handle;
	starpu_data_handle_t dro_handle;
	starpu_data_handle_t basis_handle;

	starpu_matrix_data_register(&ro_handle, 0,
				    (uintptr_t)ro,Neq_max,Neq_max,Np, sizeof(double));
	starpu_matrix_data_register(&dro_handle, 0,
				    (uintptr_t)dro,Neq_max,Neq_max,Np, sizeof(double));
	starpu_matrix_data_register(&basis_handle, 0,
				    (uintptr_t)basis,Np,Np,Ng, sizeof(double));

	*ro_h = ro_handle;
	*dro_h = dro_handle;
	*basis_h = basis_handle;
}

void starpu_unregister_element_c(void **ro_h, void **dro_h, void **basis_h)
{
	starpu_data_handle_t ro_handle = *ro_h;
	starpu_data_handle_t dro_handle = *dro_h;
	starpu_data_handle_t basis_handle = *basis_h;

	starpu_data_unregister(ro_handle);
	starpu_data_unregister(dro_handle);
	starpu_data_unregister(basis_handle);
}

//--------------------------------------------------------------//
void loop_element_cpu_fortran(double coeff, int Neq_max, int Np, int Ng, void *ro_ptr, void *dro_ptr, void *basis_ptr, void *cl_arg);

void loop_element_cpu_func(void *buffers[], void *cl_arg);

struct starpu_codelet cl_loop_element =
{
	.cpu_funcs = {loop_element_cpu_func},
	.nbuffers = 3,
	.modes = {STARPU_R,STARPU_RW,STARPU_R},
	.name = "LOOP_ELEMENT"
};

void loop_element_cpu_func(void *buffers[], void *cl_arg)
{
	double coeff;

	double **ro = (double **) STARPU_MATRIX_GET_PTR(buffers[0]);
	int Neq_max  = STARPU_MATRIX_GET_NX(buffers[0]);

	double **dro = (double **) STARPU_MATRIX_GET_PTR(buffers[1]);

	double **basis = (double **) STARPU_MATRIX_GET_PTR(buffers[2]);
	int Np = STARPU_MATRIX_GET_NX(buffers[2]);
	int Ng = STARPU_MATRIX_GET_NY(buffers[2]);

	starpu_codelet_unpack_args(cl_arg, &coeff);

	void *ro_ptr    = &ro;
	void *dro_ptr   = &dro;
	void *basis_ptr = &basis;

	loop_element_cpu_fortran(coeff,Neq_max,Np,Ng,
				 ro_ptr,dro_ptr,basis_ptr,cl_arg);
}

void starpu_loop_element_task_c(double coeff, void **ro_h, void **dro_h, void **basis_h)
{
	int ret;

	starpu_data_handle_t ro_handle = *ro_h;
	starpu_data_handle_t dro_handle = *dro_h;
	starpu_data_handle_t basis_handle = *basis_h;

	/* execute the task on any eligible computational ressource */
	ret = starpu_task_insert(&cl_loop_element,
				 STARPU_VALUE, &coeff, sizeof(double),
				 STARPU_R,     ro_handle,
				 STARPU_RW,    dro_handle,
				 STARPU_R,     basis_handle,
				 0);

	/* verification */
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_insert");
}

//--------------------------------------------------------------//
void copy_element_cpu_fortran(int Neq_max, int Np, void *ro_ptr, void *dro_ptr);

void copy_element_cpu_func(void *buffers[], void *cl_arg);

struct starpu_codelet cl_copy_element =
{
	.cpu_funcs = {copy_element_cpu_func},
	.nbuffers = 2,
	.modes = {STARPU_RW,STARPU_R},
	.name = "COPY_ELEMENT"
};

void copy_element_cpu_func(void *buffers[], void *cl_arg)
{
	(void)cl_arg;
	double **ro = (double **) STARPU_MATRIX_GET_PTR(buffers[0]);
	int Neq_max  = STARPU_MATRIX_GET_NX(buffers[0]);
	int Np = STARPU_MATRIX_GET_NY(buffers[0]);

	double **dro = (double **) STARPU_MATRIX_GET_PTR(buffers[1]);

	void *ro_ptr    = &ro;
	void *dro_ptr   = &dro;

	copy_element_cpu_fortran(Neq_max,Np,ro_ptr,dro_ptr);
}

void starpu_copy_element_task_c(void **ro_h, void **dro_h)
{
	int ret;

	starpu_data_handle_t ro_handle = *ro_h;
	starpu_data_handle_t dro_handle = *dro_h;

	/* execute the task on any eligible computational ressource */
	ret = starpu_insert_task(&cl_copy_element,
				 STARPU_RW,  ro_handle,
				 STARPU_R,   dro_handle,
				 0);

	/* verification */
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_task_submit");
}

//--------------------------------------------------------------//
int starpu_my_init_c()
{
	/* Initialize StarPU with default configuration */
	int ret;
	struct starpu_conf conf;
	starpu_conf_init(&conf);
	conf.sched_policy_name = "dmda";

	ret = starpu_init(&conf);
	/*     int ret = starpu_init(NULL); */
	return ret;
}
