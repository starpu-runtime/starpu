/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011 William Braik, Yann Courtois, Jean-Marie Couteyen, Anthony
 * Roy
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

#include <semaphore.h> 
#include <pthread.h>

#ifndef __STARPUTOP_MESSAGE_QUEUE_H__
#define __STARPUTOP_MESSAGE_QUEUE_H__

typedef struct starputop_message_queue_item
{
	char *message;
	struct starputop_message_queue_item* next;
} starputop_message_queue_item_t;

typedef struct starputop_message_queue
{
	struct starputop_message_queue_item* head;
	struct starputop_message_queue_item* tail;
	sem_t semaphore;
	pthread_mutex_t mutex;
} starputop_message_queue_t;


starputop_message_queue_t *starputop_message_add(
			starputop_message_queue_t*,
			char*);

char* starputop_message_remove(starputop_message_queue_t*);

starputop_message_queue_t* starputop_message_queue_new();
starputop_message_queue_t* starputop_message_queue_free(
			starputop_message_queue_t*);

#endif
