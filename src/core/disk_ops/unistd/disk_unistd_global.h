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

#ifndef __DISK_UNISTD_GLOBAL_H__
#define __DISK_UNISTD_GLOBAL_H__

struct starpu_unistd_global_obj {
        int descriptor;
        char * path;
        double size;
	int flags;
	starpu_pthread_mutex_t mutex;
};

 void * starpu_unistd_global_alloc (struct starpu_unistd_global_obj * obj, void *base, size_t size STARPU_ATTRIBUTE_UNUSED);
 void starpu_unistd_global_free (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED);
 void * starpu_unistd_global_open (struct starpu_unistd_global_obj * obj, void *base, void *pos, size_t size);
 void starpu_unistd_global_close (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, size_t size STARPU_ATTRIBUTE_UNUSED);
 ssize_t starpu_unistd_global_read (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, void *buf, off_t offset, size_t size, void * _starpu_aiocb_disk);
 ssize_t starpu_unistd_global_write (void *base STARPU_ATTRIBUTE_UNUSED, void *obj, const void *buf, off_t offset, size_t size, void * _starpu_aiocb_disk);
 void * starpu_unistd_global_plug (void *parameter);
 void starpu_unistd_global_unplug (void *base);
 int get_unistd_global_bandwidth_between_disk_and_main_ram(unsigned node);

#endif
