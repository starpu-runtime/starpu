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

#include <common/uthash.h>
#include <common/list.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_tracker.h>
#include "starpu_mpi_checkpoint_template.h"

struct _starpu_mpi_checkpoint_domain_tracker_index_list* domain_tracker_list;
starpu_pthread_mutex_t tracker_mutex;

struct _starpu_mpi_checkpoint_domain_tracker_entry
{
	UT_hash_handle                        hh;
	int                                   instance;
	struct _starpu_mpi_checkpoint_tracker tracker;
};

LIST_TYPE(_starpu_mpi_checkpoint_domain_tracker_index,
	  int domain;
	  struct _starpu_mpi_checkpoint_tracker* last_valid_instance;
	  struct _starpu_mpi_checkpoint_domain_tracker_entry* tracked_inst_hash_table;
)

static inline void _starpu_mpi_checkpoint_domain_tracker_index_init(struct _starpu_mpi_checkpoint_domain_tracker_index* index)
{
	index->domain = -1;
	index->tracked_inst_hash_table = NULL;
	index->last_valid_instance = NULL;
}

static inline void _starpu_mpi_checkpoint_domain_tracker_entry_init(struct _starpu_mpi_checkpoint_domain_tracker_entry* entry)
{
	entry->instance = -1;
	entry->tracker.cp_id = -1;
	entry->tracker.cp_inst = -1;
	entry->tracker.cp_domain = -1;
	entry->tracker.cp_template = NULL;
	entry->tracker.ack_msg_count = 0;
	entry->tracker.first_msg_sent_flag = 0;
	entry->tracker.valid = 0;
	entry->tracker.old = 0;
}

static inline struct _starpu_mpi_checkpoint_domain_tracker_index* get_domain_tracker_index(int domain)
{
	struct _starpu_mpi_checkpoint_domain_tracker_index* index;
	for (index = _starpu_mpi_checkpoint_domain_tracker_index_list_begin(domain_tracker_list) ;
		index != _starpu_mpi_checkpoint_domain_tracker_index_list_end(domain_tracker_list) ;
		index = _starpu_mpi_checkpoint_domain_tracker_index_list_next(index))
	{
		if (index->domain == domain)
		{
			return index;
		}
	}
	return NULL;
}

static inline struct _starpu_mpi_checkpoint_domain_tracker_index* add_domain_tracker_index(int domain)
{
	struct _starpu_mpi_checkpoint_domain_tracker_index* index;
	_STARPU_MPI_MALLOC(index, sizeof(struct _starpu_mpi_checkpoint_domain_tracker_index));
	_starpu_mpi_checkpoint_domain_tracker_index_init(index);
	index->domain = domain;
	_starpu_mpi_checkpoint_domain_tracker_index_list_push_back(domain_tracker_list, index);
	return index;
}

static inline struct _starpu_mpi_checkpoint_domain_tracker_entry* get_tracker_entry(struct _starpu_mpi_checkpoint_domain_tracker_index* index, int instance)
{
	struct _starpu_mpi_checkpoint_domain_tracker_entry* entry = NULL;
	if (index->tracked_inst_hash_table)
	{
		HASH_FIND_INT(index->tracked_inst_hash_table, &instance, entry);
	}
	return entry;
}

static inline struct _starpu_mpi_checkpoint_domain_tracker_entry* add_tracker_entry(struct _starpu_mpi_checkpoint_domain_tracker_index* index, int cp_id, int cp_inst, int cp_domain, starpu_mpi_checkpoint_template_t cp_template)
{
	struct _starpu_mpi_checkpoint_domain_tracker_entry* entry;
	_STARPU_MPI_MALLOC(entry, sizeof(struct _starpu_mpi_checkpoint_domain_tracker_entry));
	_starpu_mpi_checkpoint_domain_tracker_entry_init(entry);
	entry->instance = cp_inst;
	entry->tracker.cp_id = cp_id;
	entry->tracker.cp_inst = cp_inst;
	entry->tracker.cp_domain = cp_domain;
	entry->tracker.cp_template = cp_template;
	entry->tracker.ack_msg_count = cp_template->message_to_send_number;
	HASH_ADD_INT(index->tracked_inst_hash_table, instance, entry);
	return entry;
}

static inline int _clear_domain_tracker_index(struct _starpu_mpi_checkpoint_domain_tracker_index* index)
{
	struct _starpu_mpi_checkpoint_domain_tracker_entry* entry, *tmp;
	HASH_ITER(hh, index->tracked_inst_hash_table, entry, tmp)
	{
		HASH_DEL(index->tracked_inst_hash_table, entry);
		free(entry);
	}
	return 0;
}

static inline int _domain_tracker_delete_all()
{
	struct _starpu_mpi_checkpoint_domain_tracker_index* temp_index;
	struct _starpu_mpi_checkpoint_domain_tracker_index* index = _starpu_mpi_checkpoint_domain_tracker_index_list_begin(domain_tracker_list) ;
	while (index != _starpu_mpi_checkpoint_domain_tracker_index_list_end(domain_tracker_list))
	{
		temp_index = _starpu_mpi_checkpoint_domain_tracker_index_list_next(index);
		_clear_domain_tracker_index(index);
		_starpu_mpi_checkpoint_domain_tracker_index_list_erase(domain_tracker_list, index);
		free(index);
		index = temp_index;
	}
	return 0;
}

int _starpu_mpi_checkpoint_tracker_init()
{
	domain_tracker_list = _starpu_mpi_checkpoint_domain_tracker_index_list_new();
	STARPU_PTHREAD_MUTEX_INIT(&tracker_mutex, NULL);
	return 0;
}

int _starpu_mpi_checkpoint_tracker_shutdown()
{
	_domain_tracker_delete_all();
	STARPU_PTHREAD_MUTEX_DESTROY(&tracker_mutex);
	free(domain_tracker_list);
	return 0;
}

struct _starpu_mpi_checkpoint_tracker* _starpu_mpi_checkpoint_template_get_tracking_inst_by_id_inst(int cp_domain, int cp_inst)
{
	STARPU_PTHREAD_MUTEX_LOCK(&tracker_mutex);
	struct _starpu_mpi_checkpoint_domain_tracker_index *index = get_domain_tracker_index(cp_domain);
	if (NULL == index)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&tracker_mutex);
		return NULL;
	}
	struct _starpu_mpi_checkpoint_domain_tracker_entry *entry = get_tracker_entry(index, cp_inst);
	if (NULL == entry)
	{
		STARPU_PTHREAD_MUTEX_UNLOCK(&tracker_mutex);
		return NULL;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&tracker_mutex);
	return &entry->tracker;
}

struct _starpu_mpi_checkpoint_tracker* _starpu_mpi_checkpoint_template_create_instance_tracker(starpu_mpi_checkpoint_template_t cp_template, int cp_id, int cp_domain, int cp_inst)
{
	STARPU_PTHREAD_MUTEX_LOCK(&tracker_mutex);
	struct _starpu_mpi_checkpoint_domain_tracker_entry *entry;
	struct _starpu_mpi_checkpoint_domain_tracker_index *index = get_domain_tracker_index(cp_domain);
	if (NULL == index)
		index = add_domain_tracker_index(cp_domain);
	entry = get_tracker_entry(index, cp_inst);
	if (NULL == entry)
		entry = add_tracker_entry(index, cp_id, cp_inst, cp_domain, cp_template);
	STARPU_PTHREAD_MUTEX_UNLOCK(&tracker_mutex);
	return &entry->tracker;
}

struct _starpu_mpi_checkpoint_tracker* _starpu_mpi_checkpoint_tracker_update(starpu_mpi_checkpoint_template_t cp_template, int cp_id, int cp_domain, int cp_instance)
{
	STARPU_PTHREAD_MUTEX_LOCK(&tracker_mutex);
	struct _starpu_mpi_checkpoint_domain_tracker_entry* entry;
	struct _starpu_mpi_checkpoint_domain_tracker_index* index = get_domain_tracker_index(cp_domain);
	if (NULL == index)
		index = add_domain_tracker_index(cp_domain);
	entry = get_tracker_entry(index, cp_instance);
	if (NULL == entry)
	{
		STARPU_ASSERT_MSG(cp_template!=NULL, "Couldn't find a CP template with the cpid:%d\n", cp_id);
		entry = add_tracker_entry(index, cp_id, cp_instance, cp_domain, cp_template);
	}
	STARPU_ASSERT_MSG(entry->tracker.ack_msg_count>0, "Error. Trying to count ack message while all have already been received. id:%d, inst:%d, remaining_ack_messages:%d\n", entry->tracker.cp_id, entry->instance, entry->tracker.ack_msg_count);
	entry->tracker.ack_msg_count--;
	STARPU_PTHREAD_MUTEX_UNLOCK(&tracker_mutex);
	return &entry->tracker;
}

int _starpu_mpi_checkpoint_check_tracker(struct _starpu_mpi_checkpoint_tracker* tracker)
{
	if (tracker->valid)
	{
		return -1;
	}
	return tracker->ack_msg_count;
}

struct _starpu_mpi_checkpoint_tracker* _starpu_mpi_checkpoint_tracker_validate_instance(struct _starpu_mpi_checkpoint_tracker* tracker)
{
	STARPU_PTHREAD_MUTEX_LOCK(&tracker_mutex);
	// Here we validate a checkpoint and return the old cp info that must be discarded
	struct _starpu_mpi_checkpoint_tracker* temp_tracker;
	struct _starpu_mpi_checkpoint_domain_tracker_index* index = get_domain_tracker_index(tracker->cp_domain);
	if (NULL == index->last_valid_instance || tracker->cp_inst > index->last_valid_instance->cp_inst)
	{
		_STARPU_MPI_DEBUG(0, "The CP (id:%d - dom:%d - inst:%d) has been fully acknowledged, and is now the latest valid CP for the domain.\n", tracker->cp_id, tracker->cp_domain, tracker->cp_inst);
		// The checkpoint to validate is the newest of the domain. Update the latest CP and return the old "latest"
		temp_tracker = index->last_valid_instance;
		index->last_valid_instance = tracker;
		tracker->valid = 1;
		if (STARPU_LIKELY(temp_tracker!=NULL))
		{
			temp_tracker->old = 1;
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&tracker_mutex);
		return temp_tracker;
	}
	else
	{
		_STARPU_MPI_DEBUG(0, "The CP (id:%d - dom:%d - inst:%d) has been fully acknowledged, while a more recent one (id:%d - dom:%d - inst:%d) is already validated.\n",
				tracker->cp_id, tracker->cp_domain, tracker->cp_inst,
				index->last_valid_instance->cp_id, index->last_valid_instance->cp_domain, index->last_valid_instance->cp_inst);
		// The checkpoint to validate is older than the latest validated, just return it to discard it
		tracker->valid = 1;
		tracker->old =1;
		STARPU_PTHREAD_MUTEX_UNLOCK(&tracker_mutex);
		return tracker;
	}
}

struct _starpu_mpi_checkpoint_tracker* _starpu_mpi_checkpoint_tracker_get_last_valid_tracker(int domain)
{
	STARPU_PTHREAD_MUTEX_LOCK(&tracker_mutex);
	struct _starpu_mpi_checkpoint_domain_tracker_index* index = get_domain_tracker_index(domain);
	STARPU_PTHREAD_MUTEX_UNLOCK(&tracker_mutex);
	return index->last_valid_instance;
}
