/*
 * StarPU
 * Copyright (C) INRIA 2008-2010 (see AUTHORS file)
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
#include "stencil.h"

extern void opencl_codelet(void *descr[], __attribute__ ((unused)) void *_args);

static int verbose = 0;

static
void display_non_zero_values(TYPE *ptr, char *msg)
{
        if(verbose) {
                int x, y, z;

                for(z = 0; z < DIM; z++)
                        for(y = 0; y < DIM; y++)
                                for(x = 0; x < DIM; x++) {
                                        TYPE r = ptr[(z + 1) * SURFACE + (y + 1) * REALDIM + x + 1 + FIRST_PAD];
                                        if(r != 0.0)
                                                printf("%s[%d, %d, %d] == %f\n", msg, z, y, x, r);
                                }
        }
}

int main(int argc, char **argv)
{
        TYPE *data;                         // original data set given to device
        TYPE *results;                      // results returned from device
        TYPE C0 = 0.25;
        TYPE C1 = 0.75;
        starpu_data_handle data_handle;
        starpu_data_handle results_handle;
        starpu_data_handle C0_handle;
        starpu_data_handle C1_handle;

	starpu_init(NULL);

        // Filter args
        argv++;
        while (argc > 1) {
                if(!strcmp(*argv, "--verbose")) {
                        verbose = 1;
                } else
                        break;
                argc--; argv++;
        }

        // Fill our data set with random float values
        {
                long i, x, y, z;

                data = (TYPE *)malloc(SIZE * sizeof(TYPE));
                results = (TYPE *)malloc(SIZE * sizeof(TYPE));
                if (!data || !results) {
                        fprintf(stderr, "Malloc failed!\n");
                        return;
                }

                for(i = 0; i < SIZE; i++) {
                        data[i] = 0.0;
                        results[i] = 0.0;
                }

                z = ZBLOCK-1;
                y = YBLOCK-1;
                x = XBLOCK-1;

                data[(z + 1) * SURFACE + (y + 1) * REALDIM + x + 1 + FIRST_PAD] = 2.0;
        }

        display_non_zero_values(data, "data");

        starpu_register_vector_data(&data_handle, 0 /* home node */,
                                    (uintptr_t)data, SIZE, sizeof(TYPE));
        starpu_register_vector_data(&results_handle, 0 /* home node */,
                                    (uintptr_t)results, SIZE, sizeof(TYPE));
        starpu_register_vector_data(&C0_handle, 0 /* home node */,
                                    (uintptr_t)&C0, 1, sizeof(TYPE));
        starpu_register_vector_data(&C1_handle, 0 /* home node */,
                                    (uintptr_t)&C1, 1, sizeof(TYPE));

        _starpu_opencl_compile_source_to_opencl("examples/stencil/stencil_opencl_codelet.cl");

	starpu_codelet cl =
	{
		.where = STARPU_OPENCL,
		.opencl_func = opencl_codelet,
		.nbuffers = 4
	};

	{
		struct starpu_task *task = starpu_task_create();

                task->cl = &cl;
                task->callback_func = NULL;

		task->buffers[0].handle = data_handle;
		task->buffers[0].mode = STARPU_R;
		task->buffers[1].handle = results_handle;
		task->buffers[1].mode = STARPU_W;
		task->buffers[2].handle = C0_handle;
		task->buffers[2].mode = STARPU_R;
		task->buffers[3].handle = C1_handle;
		task->buffers[3].mode = STARPU_R;

		int ret = starpu_submit_task(task);
		if (STARPU_UNLIKELY(ret == -ENODEV))
		{
			fprintf(stderr, "No worker may execute this task\n");
			exit(0);
		}
	}

	starpu_wait_all_tasks();

	/* update the array in RAM */
	starpu_sync_data_with_mem(data_handle, STARPU_R);
	starpu_sync_data_with_mem(results_handle, STARPU_R);
	starpu_sync_data_with_mem(C0_handle, STARPU_R);
	starpu_sync_data_with_mem(C1_handle, STARPU_R);

	display_non_zero_values(results, "results");

	starpu_release_data_from_mem(data_handle);
	starpu_release_data_from_mem(results_handle);
	starpu_release_data_from_mem(C0_handle);
	starpu_release_data_from_mem(C1_handle);

	starpu_shutdown();

	return 0;
}
