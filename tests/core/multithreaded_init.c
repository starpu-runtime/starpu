/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Institut National de Recherche en Informatique et Automatique
 * Copyright (C) 2010  Université de Bordeaux 1
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
#include <sys/time.h>
#include <stdio.h>
#include <pthread.h>
#include <starpu.h>

#define NUM_THREADS 5

void *launch_starpu(void *id)
{ 
   starpu_init(NULL);
   return NULL;
}

int main(int argc, char **argv)
{ 
  unsigned i;
  double timing;
  struct timeval start;
  struct timeval end;

  pthread_t threads[NUM_THREADS];
  
  gettimeofday(&start, NULL);

  for (i = 0; i < NUM_THREADS; ++i)
    {
      int ret = pthread_create(&threads[i], NULL, launch_starpu, NULL);
      STARPU_ASSERT(ret == 0);
    }

  for (i = 0; i < NUM_THREADS; ++i)
    {
      int ret = pthread_join(threads[i], NULL);
      STARPU_ASSERT(ret == 0);
    }

  gettimeofday(&end, NULL);

  timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));

  fprintf(stderr, "Success : %d threads launching simultaneously starpu_init\n", NUM_THREADS);
  fprintf(stderr, "Total: %lf secs\n", timing/1000000);
  fprintf(stderr, "Per task: %lf usecs\n", timing/NUM_THREADS);

  starpu_shutdown();

  return 0;
}
