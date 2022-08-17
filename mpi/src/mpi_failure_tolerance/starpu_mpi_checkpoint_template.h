/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef _STARPU_MPI_CHECKPOINT_TEMPLATE_H
#define _STARPU_MPI_CHECKPOINT_TEMPLATE_H

#include <starpu_mpi.h>
#include <common/list.h>
#include <starpu_mpi_private.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_tracker.h>

#ifdef __cplusplus
extern "C"
{
#endif

#define MAX_CP_TEMPLATE_NUMBER 32 // Arbitrary limit

#define _CHECKPOINT_TEMPLATE_BACKUPED_RANK_ARRAY_DEFAULT_SIZE 2

extern starpu_pthread_mutex_t           cp_template_mutex;
extern int                              cp_template_array_size;

extern starpu_mpi_checkpoint_template_t cp_template_array[MAX_CP_TEMPLATE_NUMBER];

int increment_current_instance();
int get_current_instance();

void checkpoint_template_lib_init(void);

void checkpoint_template_lib_quit(void);

int _checkpoint_template_digest_ack_reception(int checkpoint_id, int checkpoint_instance);
void _checkpoint_template_digest_ack_reception_cb(void* _arg);
void _cp_discard_message_recv_cb(void* _args);

starpu_mpi_checkpoint_template_t _starpu_mpi_get_checkpoint_template_by_id(int checkpoint_id);
int _starpu_mpi_checkpoint_post_cp_discard_recv(starpu_mpi_checkpoint_template_t cp_template);

int _starpu_mpi_checkpoint_template_register(starpu_mpi_checkpoint_template_t *cp_template, int cp_id, int cp_domain, va_list varg_list);

LIST_TYPE(_starpu_mpi_checkpoint_template_tracking_inst,
	int cp_id;
	int                              cp_inst;
	int                              cp_domain;
	starpu_mpi_checkpoint_template_t cp_template;
	int                              ack_msg_count;
	int                              valid:1;
)

LIST_TYPE(_starpu_mpi_checkpoint_template_item,
	int type;
	void             *ptr;
	size_t           count;
	int              backupped_by;
	int              backup_of;
	starpu_mpi_tag_t tag;
)

struct _starpu_mpi_checkpoint_template
{
	struct _starpu_mpi_checkpoint_template_item_list list;
	int                                              size;
	int                                              cp_id;
	int                                              checkpoint_domain;
	int                                              message_to_send_number;
	int                                              frozen;
	starpu_pthread_mutex_t                           mutex;
	int                                              *backup_of_array;
	int                                              backup_of_array_max_size;
	int                                              backup_of_array_used_size;
	int                                              *backupped_by_array;
	int                                              backupped_by_array_max_size;
	int                                              backupped_by_array_used_size;

};

static inline int checkpoint_template_array_realloc(int** array, int* max_size, int growth_factor)
{
	//	fprintf(stderr, "old array %p - first elem %d\n", *array, *array[0]);
	//	fprintf(stderr, "Newsize=%d\n", growth_factor*(*max_size));
	_STARPU_MPI_REALLOC(*array, growth_factor*(*max_size)*sizeof(int));
	//	fprintf(stderr, "Newarray=%p\n", *array);
	*max_size = growth_factor*(*max_size);
	return *max_size;
}

static inline int checkpoint_template_backup_of_array_realloc_double(struct _starpu_mpi_checkpoint_template* checkpoint_template)
{
	return checkpoint_template_array_realloc(&checkpoint_template->backup_of_array, &checkpoint_template->backup_of_array_max_size, 2);
}

static inline int checkpoint_template_backupped_by_array_realloc_double(struct _starpu_mpi_checkpoint_template* checkpoint_template)
{
	return checkpoint_template_array_realloc(&checkpoint_template->backupped_by_array, &checkpoint_template->backupped_by_array_max_size, 2);
}

static inline struct _starpu_mpi_checkpoint_template_item* _starpu_mpi_checkpoint_template_item_create(int type, void* ptr, int count, int backupped_by, int backup_of, starpu_mpi_tag_t tag)
{
	struct _starpu_mpi_checkpoint_template_item* item;
	_STARPU_MPI_CALLOC(item, 1, sizeof(struct _starpu_mpi_checkpoint_template_item));
	item->type         = type;
	item->ptr          = ptr;
	item->count        = count;
	item->backupped_by = backupped_by;
	item->backup_of    = backup_of;
	item->tag          = tag;

	return item;
}

static inline starpu_mpi_checkpoint_template_t _starpu_mpi_checkpoint_template_new(int cp_id, int cp_domain)
{
	starpu_mpi_checkpoint_template_t _cp_template;
	_STARPU_MPI_CALLOC(_cp_template, 1, sizeof(struct _starpu_mpi_checkpoint_template));
	_cp_template->cp_id                    = cp_id;
	_cp_template->checkpoint_domain        = cp_domain;
	_cp_template->backup_of_array_max_size = _CHECKPOINT_TEMPLATE_BACKUPED_RANK_ARRAY_DEFAULT_SIZE;
	_STARPU_MPI_MALLOC(_cp_template->backup_of_array, _CHECKPOINT_TEMPLATE_BACKUPED_RANK_ARRAY_DEFAULT_SIZE);
	_cp_template->backup_of_array[0] = -1;
	_cp_template->backup_of_array_used_size = 0;
	_cp_template->backupped_by_array_max_size     = _CHECKPOINT_TEMPLATE_BACKUPED_RANK_ARRAY_DEFAULT_SIZE;
	_STARPU_MPI_MALLOC(_cp_template->backupped_by_array, _CHECKPOINT_TEMPLATE_BACKUPED_RANK_ARRAY_DEFAULT_SIZE);
	_cp_template->backupped_by_array[0] = -1;
	_cp_template->backupped_by_array_used_size = 0;
	STARPU_PTHREAD_MUTEX_INIT(&_cp_template->mutex, NULL);
	return _cp_template;
}

static inline int _checkpoint_template_add_to_backup_arrays(starpu_mpi_checkpoint_template_t cp_template, int backupped_by, int backup_of)
{
	int i;
	if (backup_of == -1)
	{
		for (i = 0; i < cp_template->backupped_by_array_used_size; i++)
		{
			if (backupped_by == cp_template->backupped_by_array[i])
			{
				return 0;
			}
		}
		if (cp_template->backupped_by_array_used_size + 1 == cp_template->backupped_by_array_max_size)
		{
			checkpoint_template_backupped_by_array_realloc_double(cp_template);
		}
		cp_template->backupped_by_array[cp_template->backupped_by_array_used_size] = backupped_by;
		cp_template->backupped_by_array_used_size++;
		cp_template->backupped_by_array[cp_template->backupped_by_array_used_size] = -1;
		return backupped_by;
	}
	else if (backupped_by == -1)
	{
		for (i = 0; i < cp_template->backup_of_array_used_size; i++)
		{
			if (backup_of == cp_template->backup_of_array[i])
			{
				return 0;
			}
		}
		if (cp_template->backup_of_array_used_size + 1 == cp_template->backup_of_array_max_size)
		{
			checkpoint_template_backup_of_array_realloc_double(cp_template);
		}
		cp_template->backup_of_array[cp_template->backup_of_array_used_size] = backup_of;
		cp_template->backup_of_array_used_size++;
		cp_template->backup_of_array[cp_template->backup_of_array_used_size] = -1;
		return backup_of;
	}
	else
	{
		_STARPU_DISP("[warning] Checkpoint template item does not refer any backup information. This should not happen.\n");
	}
	return -1;
}

static inline struct _starpu_mpi_checkpoint_template_item* _starpu_mpi_checkpoint_template_get_first_data(starpu_mpi_checkpoint_template_t template)
{
	return _starpu_mpi_checkpoint_template_item_list_front(&template->list);
}

static inline struct _starpu_mpi_checkpoint_template_item* _starpu_mpi_checkpoint_template_get_next_data(starpu_mpi_checkpoint_template_t template STARPU_ATTRIBUTE_UNUSED, struct _starpu_mpi_checkpoint_template_item* ref_data)
{
	return _starpu_mpi_checkpoint_template_item_list_next(ref_data);
}

static inline struct _starpu_mpi_checkpoint_template_item* _starpu_mpi_checkpoint_template_end(starpu_mpi_checkpoint_template_t template STARPU_ATTRIBUTE_UNUSED)
{
	return NULL;
}

static inline int _starpu_checkpoint_template_free(starpu_mpi_checkpoint_template_t cp_template)
{
	struct _starpu_mpi_checkpoint_template_item* item;
	struct _starpu_mpi_checkpoint_template_item* next_item;
	STARPU_PTHREAD_MUTEX_LOCK(&cp_template->mutex);
	item = _starpu_mpi_checkpoint_template_get_first_data(cp_template);
	while (item != _starpu_mpi_checkpoint_template_end(cp_template))
	{
		next_item = _starpu_mpi_checkpoint_template_get_next_data(cp_template, item);
		free(item);
		item = next_item;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&cp_template->mutex);
	STARPU_PTHREAD_MUTEX_DESTROY(&cp_template->mutex);
	free(cp_template);
	return 0;
}

#ifdef __cplusplus
}
#endif

#endif //_STARPU_MPI_CHECKPOINT_TEMPLATE_H
