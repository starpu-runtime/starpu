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
 * Garbage collection thread
 */

/* List of entities to be released */
static volatile entity gc_list = NULL;
static volatile entity entities = NULL;

/* Mutex and cond for release */
static pthread_mutex_t gc_mutex = PTHREAD_MUTEX_INITIALIZER;
static pthread_cond_t  gc_cond = PTHREAD_COND_INITIALIZER;

/* Set to 1 to stop release thread execution */
static volatile int gc_stop_required = 0;

#define GC_LOCK pthread_mutex_lock(&gc_mutex)
#define GC_UNLOCK { pthread_cond_signal(&gc_cond); \
                    pthread_mutex_unlock(&gc_mutex);}
#define GC_UNLOCK_NO_SIGNAL pthread_mutex_unlock(&gc_mutex)

/* Thread routine */
static void * gc_thread_routine(void *UNUSED(arg)) {
  
  GC_LOCK;

  do {
    /* Make a copy of the gc_list to allow callbacks to add things into it */
    entity rs = gc_list;
    gc_list = NULL;

    GC_UNLOCK_NO_SIGNAL;

    entity r = rs;
    while (r != NULL) {
      /* Call entity release callback */
      if (r->release_callback != NULL) {
        r->release_callback(r);
      }

      /* Release entity */
      entity next = r->next;
      free(r);

      r = next;
    }

    GC_LOCK;

    /* Check if new entities have been added */
    if (gc_list != NULL)
      continue;

    /* Stop if required */
    if (gc_stop_required) {
      GC_UNLOCK_NO_SIGNAL;
      break;
    }

    /* Otherwise we sleep */
    pthread_cond_wait(&gc_cond, &gc_mutex);

  } while (1);

  pthread_exit(NULL);
}

static pthread_t gc_thread;

/* Start garbage collection */
static void gc_start(void) {
  pthread_create(&gc_thread, NULL, gc_thread_routine, NULL);
}

/* Stop garbage collection */
static void gc_stop(void) {
  GC_LOCK;

  gc_stop_required = 1;

  GC_UNLOCK;

  pthread_join(gc_thread, NULL);
}

/**
 * Decrement reference counter and release entity if applicable
 */
#define gc_entity_release(a) gc_entity_release_ex(&(a)->_entity)

static void gc_entity_release_ex(entity e) {

  /* Decrement reference count */
  int refs = __sync_sub_and_fetch(&e->refs, 1);

  if (refs != 0)
    return;

  GC_LOCK;

  /* Remove entity from the entities list */
  if (e->prev != NULL)
    e->prev->next = e->next;
  if (e->next != NULL)
    e->next->prev = e->prev;
  if (entities == e)
    entities = e->next;

  /* Put entity in the release queue */
  e->next = gc_list;
  gc_list = e;

  GC_UNLOCK;
}



/**
 * Initialize entity
 */
static void gc_entity_init(void *arg, void (*release_callback)(void*)) {
  struct entity * e = (entity)arg;

  e->refs = 1;
  e->release_callback = release_callback;
  e->prev = NULL;

  GC_LOCK;

  e->next = entities;
  if (entities != NULL)
    entities->prev = e;
  entities = e;

  GC_UNLOCK_NO_SIGNAL;
}

/**
 * Allocate and initialize entity
 */
static void * gc_entity_alloc(unsigned int size, void (*release_callback)(void*)) {
  void * e = malloc(size);
  gc_entity_init(e, release_callback);
  return e;
}

/**
 * Retain entity
 */
static void gc_entity_retain(void *arg) {

  struct entity * e = (entity)arg;

  __sync_fetch_and_add(&e->refs, 1);
}


#define gc_entity_store(dest,e) \
  do {\
    gc_entity_retain(e); \
    *dest = e;\
  } while(0);

#define gc_entity_unstore(dest) \
  do {\
    gc_entity_release(*dest); \
    *dest = NULL;\
  } while(0);

#undef GC_LOCK
#undef GC_UNLOCK
#undef GC_UNLOCK_NO_SIGNAL
