/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2013-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <stdarg.h>
#include <stdlib.h>

#include <sys/param.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_cache.h>
#include <mpi/starpu_mpi_mpi_backend.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_template.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint.h>
#include <mpi_failure_tolerance/starpu_mpi_ft_service_comms.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_package.h>
#include <mpi_failure_tolerance/starpu_mpi_ft_stats.h>

starpu_pthread_mutex_t           cp_template_mutex;
starpu_pthread_mutex_t           current_instance_mutex;
starpu_mpi_checkpoint_template_t cp_template_array[MAX_CP_TEMPLATE_NUMBER];
int                              cp_template_array_size = 0;
static int                       my_rank;
static int                       comm_size;
static int                       current_instance;

typedef int (*backup_of_fn)(int);

int increment_current_instance()
{
	int _inst;
	STARPU_PTHREAD_MUTEX_LOCK(&current_instance_mutex);
	_inst = ++current_instance;
	STARPU_PTHREAD_MUTEX_UNLOCK(&current_instance_mutex);
	return _inst;
}

int get_current_instance()
{
	int _inst;
	STARPU_PTHREAD_MUTEX_LOCK(&current_instance_mutex);
	_inst = current_instance;
	STARPU_PTHREAD_MUTEX_UNLOCK(&current_instance_mutex);
	return _inst;
}

void checkpoint_template_lib_init(void)
{
	STARPU_PTHREAD_MUTEX_INIT(&current_instance_mutex, NULL);
	STARPU_PTHREAD_MUTEX_INIT(&cp_template_mutex, NULL);
	starpu_mpi_comm_rank(MPI_COMM_WORLD, &_my_rank);
	starpu_mpi_comm_size(MPI_COMM_WORLD, &comm_size);
	current_instance = 0;
#ifdef STARPU_MPI_VERBOSE
	_starpu_mpi_set_debug_level_max(1000);
#endif
}

void checkpoint_template_lib_quit(void)
{
	int i;
	for (i=0 ; i<MAX_CP_TEMPLATE_NUMBER ; i++)
	{
		if (cp_template_array[i] == NULL)
		{
			break;
		}
		_starpu_checkpoint_template_free(cp_template_array[i]);
		cp_template_array[i] = NULL;
	}
}

int _starpu_mpi_checkpoint_template_add_data(starpu_mpi_checkpoint_template_t cp_template, int type, void* ptr, int count, int backupped_by, int backup_of, starpu_mpi_tag_t tag)
{
	STARPU_PTHREAD_MUTEX_LOCK(&cp_template->mutex);
	STARPU_ASSERT_MSG(!cp_template->frozen, "It is not possible to modify registered checkpoint template.\n");
	struct _starpu_mpi_checkpoint_template_item* item;
	item = _starpu_mpi_checkpoint_template_item_create(type, ptr, count, backupped_by, backup_of, tag);
	_starpu_mpi_checkpoint_template_item_list_push_back(&cp_template->list, item);
	_checkpoint_template_add_to_backup_arrays(cp_template, backupped_by, backup_of);
	_STARPU_MPI_DEBUG(5, "New checkpoint data entry %p (data:%p) has been added to cp_template with id:%d. (%s)\n", item, item->ptr, cp_template->cp_id, backupped_by == -1 ? "BACKUP_OF" : "BACKUPPED_BY");
	STARPU_PTHREAD_MUTEX_UNLOCK(&cp_template->mutex);
	return 0;
}

int starpu_mpi_checkpoint_template_create(starpu_mpi_checkpoint_template_t* cp_template, int cp_id, int cp_domain)
{
	*cp_template = _starpu_mpi_checkpoint_template_new(cp_id, cp_domain);
	return 0;
}

int _starpu_mpi_checkpoint_template_add_entry(starpu_mpi_checkpoint_template_t cp_template, int arg_type, va_list varg_list)
{
	void*        ptr;
	int              count;
	int              backupped_by;
	int              data_rank;
	starpu_mpi_tag_t tag;
	backup_of_fn     _backup_of;
	int i;

	arg_type = arg_type & ~STARPU_COMMUTE;

	switch(arg_type)
	{
		case STARPU_R:
			ptr          = va_arg(varg_list, void*);
			count        = 1;
			backupped_by = va_arg(varg_list, int);
			data_rank    = starpu_mpi_data_get_rank((starpu_data_handle_t)ptr);
			if (_my_rank==data_rank)
			{
				return _starpu_mpi_checkpoint_template_add_data(cp_template, arg_type, ptr, count, backupped_by, -1, -1);
			}
			else if(_my_rank == backupped_by)
			{
				return _starpu_mpi_checkpoint_template_add_data(cp_template, arg_type, ptr, count, -1, data_rank, -1);
			}
			else
			{
				/* Since this data does not concern me (i.e. it is nor my data neither a data which I'm the back up)
				 * it is considered unnecessary to register in the CP */
				return 0;
			}
			break;
		case STARPU_VALUE:
			ptr       = va_arg(varg_list, void*);
			count     = va_arg(varg_list, int);
			tag       = va_arg(varg_list, starpu_mpi_tag_t);
			_backup_of = va_arg(varg_list, backup_of_fn);
			/* I register the backup that will save this data */
			_starpu_mpi_checkpoint_template_add_data(cp_template, arg_type, ptr, count, _backup_of(_my_rank), -1, tag);
			for (i=0 ; i<_my_rank ; i++)
			{
				if (_backup_of(i) == _my_rank)
				{
					/* I'm the back up of someone else for this data, I have to remember it */
					_starpu_mpi_checkpoint_template_add_data(cp_template, arg_type, ptr, count, -1, i, tag);
				}
			}
			for (i=_my_rank+1 ; i<comm_size ; i++)
			{
				if (_backup_of(i) == _my_rank)
				{
					/* I'm the back up of someone else for this data, I have to remember it */
					_starpu_mpi_checkpoint_template_add_data(cp_template, arg_type, ptr, count, -1, i, tag);
				}
			}
			return 0;
//			case STARPU_DATA_ARRAY:
//				ptr         = va_arg(varg_list, void*);
//				count       = va_arg(varg_list, int);
//				backupped_by = va_arg(varg_list, int);
//				backup_of   = -1;
//				break;
		default:
			STARPU_ABORT_MSG("Unrecognized argument %d, did you perhaps forget to end arguments with 0?\n", arg_type);
	}
}

void _cp_discard_message_recv_cb(void* _args)
{
	// TODO: store the information of the new CP, for restart purpose
	struct _starpu_mpi_cp_discard_arg_cb* arg = (struct _starpu_mpi_cp_discard_arg_cb*) _args;
	_STARPU_MPI_FT_STATS_RECV_FT_SERVICE_MSG(sizeof(struct _starpu_mpi_cp_ack_msg));
	_STARPU_MPI_DEBUG(0, "DISCARDING OLD CHECKPOINT DATA of rank %d - new one is CPID:%d - CPINST:%d\n", arg->rank, arg->msg.checkpoint_id, arg->msg.checkpoint_instance);
	checkpoint_package_data_del(arg->msg.checkpoint_id, arg->msg.checkpoint_instance, arg->rank);
	// TODO free _args
}


int _starpu_mpi_checkpoint_post_cp_discard_recv(starpu_mpi_checkpoint_template_t cp_template)
{
	/* A new CP is submitted. We must post matching recv for the message warning the future checkpoint integrity (so
	 * I can discard old data from deprecated checkpoint).
	 * I will receive a msg if I have old CP data.
	 * TODO: For the message logging discard, I will receive message from the people I exchanged with since the last checkpoint.
	 * */
	struct _starpu_mpi_cp_discard_arg_cb* arg;
	int                              i;

	for (i=0 ; i<cp_template->backup_of_array_used_size ; i++)
	{
		_STARPU_MPI_MALLOC(arg, sizeof(struct _starpu_mpi_cp_discard_arg_cb));
		arg->rank = cp_template->backup_of_array[i];
		_STARPU_MPI_DEBUG(10, "Post DISCARD msg reception from %d\n", arg->rank);

		_starpu_mpi_ft_service_post_special_recv(_STARPU_MPI_TAG_CP_INFO);
//		_ft_service_msg_irecv_cb(&arg->msg, sizeof(struct _starpu_mpi_cp_ack_msg), arg->rank, _STARPU_MPI_TAG_CP_INFO,
//		                         MPI_COMM_WORLD, _cp_discard_message_recv_cb, (void *) arg);
	}
	return i;
}

void _cp_discard_message_send_cb(void* _args)
{
	_STARPU_MPI_FT_STATS_SEND_FT_SERVICE_MSG(sizeof(struct _starpu_mpi_cp_ack_msg));
	free(_args);
}

int _starpu_mpi_checkpoint_post_cp_discard_send(starpu_mpi_checkpoint_template_t cp_template, int cp_id, int cp_instance)
{
	/* The CP data replication has succeeded. I must send the message warning the checkpoint integrity (so
	 * they can discard old data from deprecated checkpoint).
	 * I will send to the ones if it has old CP data from me.
	 * TODO: For the message logging discard, I will send message to the people I exchanged with since the last checkpoint.
	 * */
	struct _starpu_mpi_cp_discard_arg_cb* arg;
	int i;

	for (i=0 ; i < cp_template->backupped_by_array_used_size ; i++)
	{
		_STARPU_MPI_MALLOC(arg, sizeof(struct _starpu_mpi_cp_discard_arg_cb));
		arg->rank = cp_template->backupped_by_array[i];
		_STARPU_MPI_DEBUG(10, "Post CP DISCARD msg sending to %d\n", arg->rank);
		arg->msg.discard=1;
		arg->msg.validation=0;
		arg->msg.checkpoint_id = cp_id;
		arg->msg.checkpoint_instance = cp_instance;
		_starpu_mpi_ft_service_post_send(&arg->msg, sizeof(struct _starpu_mpi_cp_ack_msg), arg->rank,
						 _STARPU_MPI_TAG_CP_INFO, MPI_COMM_WORLD, _cp_discard_message_send_cb, (void *) arg);
	}

	return 0;
}

starpu_mpi_checkpoint_template_t _starpu_mpi_get_checkpoint_template_by_id(int checkpoint_id)
{
	int i;
	STARPU_PTHREAD_MUTEX_LOCK(&cp_template_mutex);
	for (i=0 ; i < cp_template_array_size ; i++)
	{
//		STARPU_PTHREAD_MUTEX_LOCK(&cp_template_array[i]->mutex);
		if (cp_template_array[i]->cp_id == checkpoint_id)
		{
//			STARPU_PTHREAD_MUTEX_UNLOCK(&cp_template_array[i]->mutex);
			STARPU_PTHREAD_MUTEX_UNLOCK(&cp_template_mutex);
			return cp_template_array[i];
		}
//		STARPU_PTHREAD_MUTEX_UNLOCK(&cp_template_array[i]->mutex);
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&cp_template_mutex);
	return NULL;
}

//int _starpu_mpi_checkpoint_post_cp_discard_recv(starpu_mpi_checkpoint_template_t cp_template)
//{
//	/* A new CP is submitted. We must post matching recv for the message warning the future checkpoint integrity (so
//	 * I can tag the data as CP validated, and discard old data from deprecated checkpoint).
//	 * I will receive a msg if I have old CP data, or if I am the back up for a node into the upcoming Checkpoint.
//	 * * Here the union of the different list is processed to post message reception only once.
//	 * TODO: For the message logging discard, I will receive message from the people I exchanged with since the last checkpoint.
//	 * */
//	struct _starpu_mpi_cp_discard_arg_cb* arg;
//	int i, j, flag;
//	starpu_mpi_checkpoint_template_t old_template;
//	for (i=0 ; i<cp_template->backup_of_array_used_size ; i++)
//	{
//		STARPU_MPI_MALLOC(arg, sizeof(struct _starpu_mpi_cp_discard_arg_cb));
//		arg->rank = cp_template->backup_of_array[i];
//		_STARPU_MPI_DEBUG(10, "Posting DISCARD msg reception from %d\n", arg->rank);
//		_ft_service_msg_irecv_cb(&arg->msg, sizeof(struct _starpu_mpi_cp_ack_msg), arg->rank, _STARPU_MPI_TAG_CP_INFO, MPI_COMM_WORLD, _cp_discard_message_recv_cb, (void*)arg);
//	}
//	if (last_valid_checkpoint.checkpoint_id == -1)
//	{
//		return -1;
//	}
//	else if (last_valid_checkpoint.checkpoint_id!=cp_template->cp_id)
//	{
//		old_template = _starpu_mpi_get_checkpoint_template_by_id(last_valid_checkpoint.checkpoint_id);
//		for (i=0 ; i<old_template->backup_of_array_used_size ; i++)
//		{
//			flag=0;
//			for(j=0 ; j<cp_template->backup_of_array_used_size ; j++)
//			{
//				if (cp_template->backup_of_array[j] == old_template->backup_of_array[i])
//				{
//					flag = 1;
//					break;
//				}
//			}
//			if (flag==0)
//			{
//				STARPU_MPI_MALLOC(arg, sizeof(struct _starpu_mpi_cp_discard_arg_cb));
//				arg->rank = old_template->backup_of_array[i];
//				_STARPU_MPI_DEBUG(10, "Posting DISCARD msg reception from %d - LAST VALIDATED CP\n", arg->rank);
//				_ft_service_msg_irecv_cb(&arg->msg, sizeof(struct _starpu_mpi_cp_ack_msg), arg->rank, _STARPU_MPI_TAG_CP_INFO, MPI_COMM_WORLD, _cp_discard_message_recv_cb, (void*)arg);
//			}
//		}
//	}
//	return 0;
//}

//int _starpu_mpi_checkpoint_post_cp_discard_send(starpu_mpi_checkpoint_template_t cp_template, int cp_id, int cp_instance)
//{
//	/* The CP data replication has succeeded. I must send the message warning the future checkpoint integrity (so
//	 * they can tag the data as CP validated, and discard old data from deprecated checkpoint).
//	 * I will send to one if it has old CP data from me, or if it is my backup for a data into the just succeeded Checkpoint.
//	 * * Here the union of the different list is processed to send message only once.
//	 * TODO: For the message logging discard, I will send message to the people I exchanged with since the last checkpoint.
//	 * */
//	struct _starpu_mpi_cp_discard_arg_cb* arg;
//	int i, j, flag;
//	starpu_mpi_checkpoint_template_t old_template;
//	for (i=0 ; i<cp_template->backupped_by_array_used_size ; i++)
//	{
//		STARPU_MPI_MALLOC(arg, sizeof(struct _starpu_mpi_cp_discard_arg_cb));
//		arg->rank = cp_template->backupped_by_array[i];
//		_STARPU_MPI_DEBUG(10, "Sending DISCARD msg reception to %d\n", arg->rank);
//		arg->msg.checkpoint_id = cp_id;
//		arg->msg.checkpoint_instance = cp_instance;
//		_ft_service_msg_isend_cb(&arg->msg, sizeof(struct _starpu_mpi_cp_ack_msg), arg->rank, _STARPU_MPI_TAG_CP_INFO, MPI_COMM_WORLD, _cp_discard_message_send_cb, (void*)arg);
//	}
//	if (last_valid_checkpoint.checkpoint_id == -1)
//	{
//		return -1;
//	}
//	else if (last_valid_checkpoint.checkpoint_id!=cp_template->cp_id)
//	{
//		old_template = _starpu_mpi_get_checkpoint_template_by_id(last_valid_checkpoint.checkpoint_id);
//		for (i=0 ; i<old_template->backupped_by_array_used_size ; i++)
//		{
//			flag=0;
//			for(j=0 ; j<cp_template->backupped_by_array_used_size ; j++)
//			{
//				if (cp_template->backupped_by_array[j] == old_template->backupped_by_array[i])
//				{
//					flag = 1;
//					break;
//				}
//			}
//			if (flag==0)
//			{
//				STARPU_MPI_MALLOC(arg, sizeof(struct _starpu_mpi_cp_discard_arg_cb));
//				arg->rank = old_template->backupped_by_array[i];
//				_STARPU_MPI_DEBUG(10, "Sending DISCARD msg to %d - OLD CP\n", arg->rank);
//				arg->msg.checkpoint_id = cp_id;
//				arg->msg.checkpoint_instance = cp_instance;
//				_ft_service_msg_isend_cb(&arg->msg, sizeof(struct _starpu_mpi_cp_ack_msg), arg->rank, _STARPU_MPI_TAG_CP_INFO, MPI_COMM_WORLD, _cp_discard_message_send_cb, (void*)arg);
//			}
//		}
//	}
//	return 0;
//}

int _starpu_mpi_checkpoint_template_freeze(starpu_mpi_checkpoint_template_t cp_template)
{
//	char str[256];
	int i;
	STARPU_PTHREAD_MUTEX_LOCK(&cp_template->mutex);
	_STARPU_MPI_DEBUG(2, "Start freezing checkpoint id:%d\n", cp_template->cp_id);
	cp_template->frozen                 = 1;
	cp_template->message_to_send_number = 0;
	cp_template->size                   = _starpu_mpi_checkpoint_template_item_list_size(&cp_template->list);

	struct _starpu_mpi_checkpoint_template_item* item = _starpu_mpi_checkpoint_template_get_first_data(cp_template);

	while (item != _starpu_mpi_checkpoint_template_end(cp_template))
	{
		if (item->backup_of==-1 && item->backupped_by!=-1)
		{
			cp_template->message_to_send_number++;
		}
		item = _starpu_mpi_checkpoint_template_get_next_data(cp_template, item);
	}
//	sprintf(str, "backupped by Array maxsize:%d - currentsize:%d - ", cp_template->backupped_by_array_max_size, cp_template->backupped_by_array_used_size);
//	for (int i=0 ; i<cp_template->backupped_by_array_used_size ; i++)
//	{
//		sprintf(str,"%s%d ", str, cp_template->backupped_by_array[i]);
//	}
//	fprintf(stderr, "%s\n", str);
//
//	sprintf(str,"backup of Array maxsize:%d - currentsize:%d - ", cp_template->backup_of_array_max_size, cp_template->backup_of_array_used_size);
//	for (int i=0 ; i<cp_template->backup_of_array_used_size ; i++)
//	{
//		sprintf(str,"%s%d ", str, cp_template->backup_of_array[i]);
//	}
//	fprintf(stderr, "%s\n", str);

	STARPU_PTHREAD_MUTEX_UNLOCK(&cp_template->mutex);

	STARPU_PTHREAD_MUTEX_LOCK(&cp_template_mutex);
	for (i=0 ; i < cp_template_array_size ; i++)
	{
		STARPU_ASSERT_MSG(cp_template_array[i]->cp_id != cp_template->cp_id, "A checkpoint with id %d has already been registered.\n", cp_template->cp_id);
	}
	cp_template_array[cp_template_array_size] = cp_template;
	cp_template_array_size++;
	STARPU_PTHREAD_MUTEX_UNLOCK(&cp_template_mutex);

	_STARPU_MPI_DEBUG(2, "Checkpoint id:%d is frozen and registered.\n", cp_template->cp_id);
	return cp_template->size;
}

int _starpu_mpi_checkpoint_template_register(starpu_mpi_checkpoint_template_t* cp_template, int cp_id, int cp_domain, va_list varg_list)
{
	int arg_type;

	starpu_mpi_checkpoint_template_t _cp_template = _starpu_mpi_checkpoint_template_new(cp_id, cp_domain);

	va_list varg_list_copy;
	va_copy(varg_list_copy, varg_list);

	while ((arg_type = va_arg(varg_list_copy, int)) != 0)
	{
		_starpu_mpi_checkpoint_template_add_entry(_cp_template, arg_type, varg_list_copy);
	}
	va_end(varg_list_copy);

	_starpu_mpi_checkpoint_template_freeze(_cp_template);

	*cp_template = _cp_template;

	return 0;
}

int starpu_mpi_checkpoint_template_freeze(starpu_mpi_checkpoint_template_t* cp_template)
{
	return _starpu_mpi_checkpoint_template_freeze(*cp_template);
}

int starpu_mpi_checkpoint_template_register(starpu_mpi_checkpoint_template_t* cp_template, int cp_id, int cp_domain, ...)
{
	va_list varg_list;
	va_start(varg_list, cp_domain);
	int ret = _starpu_mpi_checkpoint_template_register(cp_template, cp_id, cp_domain, varg_list);
	va_end(varg_list);
	return ret;
}

int starpu_mpi_checkpoint_template_add_entry(starpu_mpi_checkpoint_template_t* cp_template, ...)
{
	va_list varg_list;
	int arg_type;
	int ret;
	va_start(varg_list, cp_template);
	arg_type = va_arg(varg_list, int);
	STARPU_ASSERT_MSG(arg_type!=STARPU_NONE, "Unhandled arg_type: STARPU_NONE(0).\n");
	ret = _starpu_mpi_checkpoint_template_add_entry(*cp_template, arg_type, varg_list);
	va_end(varg_list);
	return ret;
}

int _checkpoint_template_digest_ack_reception(int checkpoint_id, int checkpoint_instance)
{
	int remaining_ack_messages;
	struct _starpu_mpi_checkpoint_tracker* tracker, *tracker1;
	starpu_mpi_checkpoint_template_t cp_template = _starpu_mpi_get_checkpoint_template_by_id(checkpoint_id);
	STARPU_PTHREAD_MUTEX_LOCK(&cp_template_mutex);
	_STARPU_MPI_DEBUG(20, "Digesting ack recv: id=%d, inst=%d\n", checkpoint_id, checkpoint_instance);

	tracker = _starpu_mpi_checkpoint_tracker_update(cp_template, checkpoint_id, cp_template->checkpoint_domain, checkpoint_instance);
	remaining_ack_messages = _starpu_mpi_checkpoint_check_tracker(tracker);

	if (remaining_ack_messages>0)
	{
		_STARPU_MPI_DEBUG(20, "The CP (id:%d - inst:%d) found, remaining ack msg awaited:%d.\n", checkpoint_id,
		                  checkpoint_instance, remaining_ack_messages);
	}
	else if (remaining_ack_messages==0)
	{
		_STARPU_MPI_DEBUG(0, "The CP (id:%d - inst:%d) has been successfully saved and acknowledged.\n", checkpoint_id, checkpoint_instance);
		tracker = _starpu_mpi_checkpoint_tracker_validate_instance(tracker);
		_STARPU_MPI_TRACE_CHECKPOINT_END(checkpoint_instance, cp_template->checkpoint_domain);
		if (tracker==NULL)
		{
			// TODO:should warn some people, because the msg logging is not implemented(this precise nodes to contact)
			_STARPU_MPI_DEBUG(0, "No previous checkpoint to discard\n");
		}
		else
		{
			if (tracker->old)
			{
				tracker1 = _starpu_mpi_checkpoint_tracker_get_last_valid_tracker(tracker->cp_domain);
				_starpu_mpi_checkpoint_post_cp_discard_send(tracker->cp_template, tracker1->cp_id, tracker1->cp_inst);
			}
			else
			{
				_starpu_mpi_checkpoint_post_cp_discard_send(tracker->cp_template, checkpoint_id, checkpoint_instance);
			}
		}
	}
	else if (remaining_ack_messages==-1)
	{
		STARPU_ABORT_MSG("Inst (id:%d - inst:%d) is already valid. should not have received an ack msg.\n", checkpoint_id, checkpoint_instance);
	}
	else
	{
		STARPU_ABORT_MSG("Critical error, can not identify %d as remaining messages\n", remaining_ack_messages);
	}

	_STARPU_MPI_DEBUG(20, "Digested\n");
	STARPU_PTHREAD_MUTEX_UNLOCK(&cp_template_mutex);
	return 0;
}

void _checkpoint_template_digest_ack_reception_cb(void* _arg)
{
	struct _starpu_mpi_cp_ack_arg_cb* arg = (struct _starpu_mpi_cp_ack_arg_cb*) _arg;
	_checkpoint_template_digest_ack_reception(arg->msg.checkpoint_id, arg->msg.checkpoint_instance);
}

// For test purpose
int starpu_mpi_checkpoint_template_print(starpu_mpi_checkpoint_template_t cp_template)
{
//	int val;
	int i = 0;
	struct _starpu_mpi_checkpoint_template_item* item = _starpu_mpi_checkpoint_template_get_first_data(cp_template);

	while (item != _starpu_mpi_checkpoint_template_end(cp_template))
	{
		fprintf(stderr,"Item %2d: ", i);
		if (item->type == STARPU_VALUE)
		{
//			fprintf(stderr, "STARPU_VALUE - Value=%d - backupof:%d - backupedby:%d\n", (*(int *)(item->ptr)), item->backup_of, item->backupped_by);
			fprintf(stderr, "STARPU_VALUE - pointer:%p - backupof:%d - backupedby:%d\n", item->ptr, item->backup_of, item->backupped_by);
		}
		else if (item->type == STARPU_R)
		{
//			val = *(int*)starpu_data_handle_to_pointer(*(starpu_data_handle_t*)(item->ptr), 0);
//			fprintf(stderr, "STARPU_R - Value=%d - backupof:%d - backupedby:%d\n", val, item->backup_of, item->backupped_by);
			fprintf(stderr, "STARPU_R - pointer:%p - backupof:%d - backupedby:%d\n", item->ptr, item->backup_of, item->backupped_by);
		}
		else if (item->type == STARPU_DATA_ARRAY)
		{
//			fprintf(stderr, "STARPU_DATA_ARRAY - Multiple values: %d", *(int*)starpu_data_handle_to_pointer(((starpu_data_handle_t)item->ptr), 0));
//
//			for (int j=1 ; j<MIN(item->count, 5) ; j++)
//			{
//				fprintf(stderr, ", %d", *(int*)starpu_data_handle_to_pointer(((starpu_data_handle_t*)item->ptr)[j], 0)); //j*sizeof(starpu_data_handle_t)
//			}
//			fprintf(stderr, "...\n");
		}
		else
		{
			printf("Unrecognized type.\n");
		}

		item = _starpu_mpi_checkpoint_template_get_next_data(cp_template, item);
		i++;
	}
	return 0;
}
