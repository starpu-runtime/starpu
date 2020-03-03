/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef SOCL_GC_H
#define SOCL_GC_H

#include "socl.h"

void gc_start(void);
void gc_stop(void);

void gc_entity_init(void *arg, void (*release_callback)(void*), char*name);

void * gc_entity_alloc(unsigned int size, void (*release_callback)(void*), char * name);

void gc_entity_retain_ex(void *arg, const char *);
#define gc_entity_retain(arg) gc_entity_retain_ex(arg, __starpu_func__)

/** Decrement reference counter and release entity if applicable */
int gc_entity_release_ex(entity e, const char*);

int gc_active_entity_count(void);
void gc_print_remaining_entities(void);

#define gc_entity_release(a) gc_entity_release_ex(&(a)->_entity, __starpu_func__)

#define gc_entity_store(dest,e) \
  do {\
    void * _e = e;\
    gc_entity_retain(_e); \
    *dest = _e;\
  } while(0);

#define gc_entity_unstore(dest) \
  do {\
    gc_entity_release(*dest); \
    *dest = NULL;\
  } while(0);



#endif
