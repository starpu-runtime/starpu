/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
struct starpu_codelet cl =
{
	.where = STARPU_CPU,
	/* => .modes = { STARPU_R, STARPU_W }, */
	.modes[1] = STARPU_W,
	.modes[0] = STARPU_R,
	.cpu_func = foo
};


static void
foo(void)
{
	struct starpu_codelet cl =
	{
		.where = STARPU_CPU,
		/* .modes = { STARPU_R, STARPU_RW, STARPU_W } */
		.modes[2] = STARPU_W,
		.modes[1] = STARPU_RW,
		.modes[0] = STARPU_R
	};
}
