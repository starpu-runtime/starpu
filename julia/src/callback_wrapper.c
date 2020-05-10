/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

void  *dummy_function_list[] = {
				starpu_matrix_filter_vertical_block,
				starpu_matrix_filter_block,
				starpu_vector_filter_block,
				starpu_init,
};

void julia_callback_func(void *user_data)
{
  volatile int *signal = (int *) user_data;

  // wakeup callback
  *(signal) = 1;

  // Wait for callback to end.
  while ((*signal) != 0);
}

void julia_wait_signal(volatile int *signal)
{
  while ((*signal) == 0);
}
