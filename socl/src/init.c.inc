/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010,2011 University of Bordeaux
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

/**
 * Initialize SOCL
 */
__attribute__((constructor)) static void socl_init() {
  
  mem_object_init();

  starpu_init(NULL);
  
  //Disable dataflow implicit dependencies
  starpu_data_set_default_sequential_consistency_flag(0);

  gc_start();
}

/**
 * Shutdown SOCL
 */
__attribute__((destructor)) static void socl_shutdown() {

  starpu_task_wait_for_all();

  gc_stop();

  starpu_task_wait_for_all();

  int i = 0;
  while (entities != NULL) {
    i++;
    entities = entities->next;
  }

  if (i != 0)
    fprintf(stderr, "Unreleased entities: %d\n", i);

  starpu_shutdown();
}
