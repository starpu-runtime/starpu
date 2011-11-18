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

#include  "starpu_top_message_queue.h"
#include  <string.h>
#include  <stdio.h>
#include  <stdlib.h>

//this global queue is used both by API and by network threads
starpu_top_message_queue_t*  starpu_top_mt = NULL;


/* Will always return the pointer to starpu_top_message_queue */
starpu_top_message_queue_t* starpu_top_message_add(
			starpu_top_message_queue_t* s,
			char* msg)
{
	starpu_top_message_queue_item_t* p = (starpu_top_message_queue_item_t *) malloc( 1 * sizeof(*p) );
	pthread_mutex_lock(&(s->mutex));
	if( NULL == p )
	{
		fprintf(stderr, "IN %s, %s: malloc() failed\n", __FILE__, "list_add");
		pthread_mutex_unlock(&(s->mutex));
		return s;
	}

	p->message = msg;
	p->next = NULL;

	if( NULL == s )
	{
		printf("Queue not initialized\n");
		pthread_mutex_unlock(&(s->mutex));
		return s;
	}
	else if( NULL == s->head && NULL == s->tail )
	{
		/* printf("Empty list, adding p->num: %d\n\n", p->num);  */
		sem_post(&(s->semaphore));
		s->head = s->tail = p;
		pthread_mutex_unlock(&(s->mutex));
		return s;
	}
	else
	{
		/* printf("List not empty, adding element to tail\n"); */
		sem_post(&(s->semaphore));
		s->tail->next = p;
		s->tail = p;
	}
	pthread_mutex_unlock(&(s->mutex));
	return s;
}

//this is a queue and it is FIFO, so we will always remove the first element
char* starpu_top_message_remove(starpu_top_message_queue_t* s)
{
	sem_wait(&(s->semaphore));
	starpu_top_message_queue_item_t* h = NULL;
	starpu_top_message_queue_item_t* p = NULL;

	if( NULL == s )
	{
		printf("List is null\n");
		return NULL;
	}
	pthread_mutex_lock(&(s->mutex));
	h = s->head;
	p = h->next;
	char* value = h->message;
	free(h);
	s->head = p;

	
	if( NULL == s->head )
		//the element tail was pointing to is free(), so we need an update
		s->tail = s->head;
	pthread_mutex_unlock(&(s->mutex));
	return value;
}


starpu_top_message_queue_t* starpu_top_message_queue_new(void)
{
	starpu_top_message_queue_t* p = (starpu_top_message_queue_t *) malloc( 1 * sizeof(*p));
	if( NULL == p )
	{
		fprintf(stderr, "LINE: %d, malloc() failed\n", __LINE__);
		return NULL;
	}

	p->head = p->tail = NULL;
	sem_init(&(p->semaphore),0,0);
	pthread_mutex_init(&(p->mutex), NULL);
	return p;
}
