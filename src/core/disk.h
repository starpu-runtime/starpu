/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013 Corentin Salingue
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

typedef void * (*disk_function)(void *, unsigned);

/* list of functions to use on disk */
struct disk_ops {
	disk_function alloc;
	disk_function free;
	disk_function read;
	disk_function write;
	disk_function open;
};

unsigned
starpu_disk_register(char * src, struct disk_ops * func);

void
starpu_disk_free(unsigned node);
