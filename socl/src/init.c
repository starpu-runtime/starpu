/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2021  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2012       Vincent Danjean
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

#include <stdlib.h>
#include "socl.h"
#include "gc.h"
#include "mem_objects.h"

int _starpu_init_failed;
volatile int _starpu_init = 0;
static starpu_pthread_mutex_t _socl_mutex = STARPU_PTHREAD_MUTEX_INITIALIZER;
static struct starpu_conf conf;

void socl_init_starpu(void) {
  STARPU_PTHREAD_MUTEX_LOCK(&_socl_mutex);
  if( ! _starpu_init ){
    starpu_conf_init(&conf);
    conf.ncuda = 0;
    conf.ncpus = 0;
    setenv("STARPU_NCUDA", "0", 1);
    setenv("STARPU_NCPUS", "0", 1);

    _starpu_init_failed = starpu_init(&conf);
    if (_starpu_init_failed != 0)
    {
       DEBUG_MSG("Error when calling starpu_init: %d\n", _starpu_init_failed);
    }
    else {
       if (starpu_opencl_worker_get_count() == 0)
       {
	    DEBUG_MSG("StarPU didn't find any OpenCL device. Try disabling CUDA support in StarPU (export STARPU_NCUDA=0).\n");
	    _starpu_init_failed = -ENODEV;
       }
    }

    /* Disable dataflow implicit dependencies */
    starpu_data_set_default_sequential_consistency_flag(0);
    _starpu_init = 1;
  }
  STARPU_PTHREAD_MUTEX_UNLOCK(&_socl_mutex);

}
/**
 * Initialize SOCL
 */
__attribute__((constructor)) static void socl_init() {


  mem_object_init();

  gc_start();
}


void soclShutdown() {
   static int shutdown = 0;

   if (!shutdown) {
      shutdown = 1;

      STARPU_PTHREAD_MUTEX_LOCK(&_socl_mutex);
      if( _starpu_init )
         starpu_task_wait_for_all();

      gc_stop();

      if( _starpu_init )
         starpu_task_wait_for_all();

      int active_entities = gc_active_entity_count();

      if (active_entities != 0) {
         DEBUG_MSG("Unreleased entities: %d\n", active_entities);
         gc_print_remaining_entities();
      }

      if( _starpu_init && _starpu_init_failed != -ENODEV)
         starpu_shutdown();
      STARPU_PTHREAD_MUTEX_UNLOCK(&_socl_mutex);

      if (socl_devices != NULL) {
         free(socl_devices);
         socl_devices = NULL;
      }
   }
}

/**
 * Shutdown SOCL
 */
__attribute__((destructor)) static void socl_shutdown() {

  char * skip_str = getenv("SOCL_SKIP_DESTRUCTOR");
  int skip = (skip_str != NULL ? atoi(skip_str) : 0);

  if (!skip) soclShutdown();

}
