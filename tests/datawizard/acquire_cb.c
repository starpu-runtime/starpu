/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

/*
 * Test that when using starpu_data_acquire_cb, the callback_w is properly called
 */

unsigned token = 0;
starpu_data_handle_t token_handle;

static
void callback_w(void *arg)
{
	(void)arg;
	token = 42;
        starpu_data_release(token_handle);
}

static
void callback_r(void *arg)
{
	(void)arg;
        starpu_data_release(token_handle);
}

int main(int argc, char **argv)
{
	int ret;

        ret = starpu_initialize(NULL, &argc, &argv);
	if (ret == -ENODEV) return STARPU_TEST_SKIPPED;
	STARPU_CHECK_RETURN_VALUE(ret, "starpu_init");

	starpu_variable_data_register(&token_handle, -1, 0, sizeof(unsigned));
	starpu_data_acquire_cb(token_handle, STARPU_W, callback_w, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_unregister(token_handle);
	STARPU_ASSERT(token == 42);

	token = 0;

	starpu_variable_data_register(&token_handle, -1, 0, sizeof(unsigned));
	starpu_data_acquire(token_handle, STARPU_W);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_release(token_handle);
	starpu_data_unregister(token_handle);

	token = 0;

	starpu_variable_data_register(&token_handle, STARPU_MAIN_RAM, (uintptr_t)&token, sizeof(unsigned));
	/* These are getting executed immediately */
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_W, callback_w, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_W, callback_w, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_RW, callback_w, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_RW, callback_w, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);

	starpu_data_acquire(token_handle, STARPU_W);
	/* These will wait for our relase */
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_W, callback_w, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_W, callback_w, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_RW, callback_w, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_RW, callback_w, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_acquire_cb(token_handle, STARPU_R, callback_r, NULL);
	starpu_data_release(token_handle);

	starpu_data_unregister(token_handle);

        FPRINTF(stderr, "Token: %u\n", token);

	starpu_shutdown();

	return (token == 42) ? EXIT_SUCCESS : EXIT_FAILURE;
}
