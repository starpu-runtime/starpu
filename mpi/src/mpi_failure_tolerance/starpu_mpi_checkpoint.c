/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2014-2022  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <common/utils.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_template.h>
#include <mpi_failure_tolerance/starpu_mpi_checkpoint_package.h>
#include <mpi_failure_tolerance/starpu_mpi_ft_service_comms.h>
#include <mpi_failure_tolerance/starpu_mpi_ft_stats.h>
#include <starpu_mpi_private.h>
#include <starpu_mpi_cache.h>
#include <mpi/starpu_mpi_mpi_backend.h> // Should be deduced at preprocessing (Nmad vs MPI)
#include "starpu_mpi_cache.h"

#define MAX_CP_TEMPLATE_NUMBER 32 // Arbitrary limit

starpu_pthread_mutex_t cp_lib_mutex;

void _ack_msg_send_cb(void* _args)
{
	struct _starpu_mpi_cp_ack_arg_cb* arg = (struct _starpu_mpi_cp_ack_arg_cb*) _args;
	_STARPU_MPI_FT_STATS_SEND_FT_SERVICE_MSG(sizeof(struct _starpu_mpi_cp_ack_msg));
	_STARPU_MPI_DEBUG(3, "Ack send succeeded cpid:%d, cpinst:%d, dest:%d\n", arg->msg.checkpoint_id, arg->msg.checkpoint_instance, arg->rank);
	//free(arg);
}

void _ack_msg_recv_cb(void* _args)
{
	struct _starpu_mpi_cp_ack_arg_cb* arg = (struct _starpu_mpi_cp_ack_arg_cb*) _args;
	int ret;
	_STARPU_MPI_FT_STATS_RECV_FT_SERVICE_MSG(sizeof(struct _starpu_mpi_cp_ack_msg));
	_STARPU_MPI_DEBUG(3, "ack msg recved id:%d inst:%d\n", arg->msg.checkpoint_id, arg->msg.checkpoint_instance);
	ret = _checkpoint_template_digest_ack_reception(arg->msg.checkpoint_id, arg->msg.checkpoint_instance);
	if (ret == 0)
	{
		//free(arg);
	}
	else if (ret == -1)
	{
		STARPU_ABORT_MSG("Could not find CP template, cpid:%d - cpinst:%d\n", arg->msg.checkpoint_id, arg->msg.checkpoint_instance);
	}
}

void _starpu_mpi_store_data_and_send_ack_cb(struct _starpu_mpi_cp_ack_arg_cb* arg)
{
	checkpoint_package_data_add(arg->msg.checkpoint_id, arg->msg.checkpoint_instance, arg->rank, arg->tag, arg->type, arg->copy_handle, arg->count);
	_STARPU_MPI_DEBUG(3,"Send ack msg to %d: id=%d inst=%d\n", arg->rank, arg->msg.checkpoint_id, arg->msg.checkpoint_instance);
	_starpu_mpi_ft_service_post_send((void *) &arg->msg, sizeof(struct _starpu_mpi_cp_ack_msg), arg->rank,
					 _STARPU_MPI_TAG_CP_ACK, MPI_COMM_WORLD, _ack_msg_send_cb, arg);
}

void _starpu_mpi_push_cp_ack_recv_cb(struct _starpu_mpi_cp_ack_arg_cb* arg)
{
	_STARPU_MPI_DEBUG(3, "Posting ack recv cb from %d\n", arg->rank);
	_starpu_mpi_ft_service_post_special_recv(_STARPU_MPI_TAG_CP_ACK);
//	_ft_service_msg_irecv_cb((void *) &arg->msg, sizeof(struct _starpu_mpi_cp_ack_msg), arg->rank,
//	                         _STARPU_MPI_TAG_CP_ACK, MPI_COMM_WORLD, _ack_msg_recv_cb, arg);
}

void _recv_internal_dup_ro_cb(void* _args)
{
	struct _starpu_mpi_cp_ack_arg_cb* arg = (struct _starpu_mpi_cp_ack_arg_cb*) _args;
	starpu_data_release(arg->copy_handle);
	_starpu_mpi_store_data_and_send_ack_cb(arg);
}

void _recv_cp_external_data_cb(void* _args)
{
	struct _starpu_mpi_cp_ack_arg_cb* arg = (struct _starpu_mpi_cp_ack_arg_cb*) _args;
	_STARPU_MPI_FT_STATS_RECV_CP_DATA(starpu_data_get_size(arg->handle));
	// an handle has specifically been created, Let's get the value back, and unregister the handle
	arg->copy_handle = starpu_data_handle_to_pointer(arg->handle, STARPU_MAIN_RAM);
	starpu_data_unregister_submit(arg->handle);
	_starpu_mpi_store_data_and_send_ack_cb(arg);
}

void _send_cp_external_data_cb(void* _args)
{
	struct _starpu_mpi_cp_ack_arg_cb* arg = (struct _starpu_mpi_cp_ack_arg_cb*) _args;
	_STARPU_MPI_FT_STATS_SEND_CP_DATA(starpu_data_get_size(arg->handle));
	free(starpu_data_handle_to_pointer(arg->handle, STARPU_MAIN_RAM));
	starpu_data_unregister_submit(arg->handle);
	_starpu_mpi_push_cp_ack_recv_cb(arg);
}

void _send_cp_internal_data_cb(void* _args)
{
	struct _starpu_mpi_cp_ack_arg_cb* arg = (struct _starpu_mpi_cp_ack_arg_cb*) _args;
	_starpu_mpi_push_cp_ack_recv_cb(_args);
	if (!arg->cache_flag)
	{
		//TODO: check cp_domain!
		struct _starpu_mpi_checkpoint_tracker* tracker = _starpu_mpi_checkpoint_template_get_tracking_inst_by_id_inst(0, arg->checkpoint_instance_hint);
		if(!tracker->first_msg_sent_flag)
		{
			tracker->first_msg_sent_flag = 1;
			_STARPU_MPI_TRACE_CHECKPOINT_BEGIN(arg->checkpoint_instance_hint,0);
		}
	}
}

void _send_internal_data_stats(struct _starpu_mpi_cp_ack_arg_cb* arg)
{
	if (arg->cache_flag)
	{
		_STARPU_MPI_FT_STATS_SEND_CACHED_CP_DATA(starpu_data_get_size(arg->handle));
	}
	else
	{
		_STARPU_MPI_FT_STATS_SEND_CP_DATA(starpu_data_get_size(arg->handle));
	}
}

int starpu_mpi_checkpoint_template_submit(starpu_mpi_checkpoint_template_t cp_template, int prio)
{
	starpu_data_handle_t handle;
	struct _starpu_mpi_data* mpi_data;
	struct _starpu_mpi_cp_ack_arg_cb* arg;
	void* cpy_ptr;
	struct _starpu_mpi_checkpoint_template_item* item;
	int current_instance;

	current_instance = increment_current_instance();
	_starpu_mpi_checkpoint_post_cp_discard_recv(cp_template);
	_starpu_mpi_checkpoint_template_create_instance_tracker(cp_template, cp_template->cp_id, cp_template->checkpoint_domain, current_instance);
	//TODO check what happens when all the ack msg are received when we arrrive here.
	item = _starpu_mpi_checkpoint_template_get_first_data(cp_template);
	while (item != _starpu_mpi_checkpoint_template_end(cp_template))
	{
		switch (item->type)
		{
			case STARPU_VALUE:
				// TODO: Maybe do not pass via starpu handles for external data, and need to reimplement mpi comm layer for
				_STARPU_MALLOC(arg, sizeof(struct _starpu_mpi_cp_ack_arg_cb));
				arg->tag = item->tag;
				arg->type = STARPU_VALUE;
				arg->count = item->count;
				arg->cache_flag = 0;
				if (item->backupped_by != -1)
				{
					_STARPU_MALLOC(cpy_ptr, item->count);
					memcpy(cpy_ptr, item->ptr, item->count);
					starpu_variable_data_register(&arg->handle, STARPU_MAIN_RAM, (uintptr_t)cpy_ptr, item->count);
					arg->rank = item->backupped_by;
					_STARPU_MPI_DEBUG(0, "Submit CP: sending external data:%d, tag:%ld, to :%d\n", (int)(*(int*)cpy_ptr), arg->tag, arg->rank);
					starpu_mpi_isend_detached_prio(arg->handle, arg->rank, arg->tag, prio, MPI_COMM_WORLD,
												   &_send_cp_external_data_cb, (void*)arg);
					// The callback needs to free the handle specially created for the send, and post ack recv
				}
				else if (item->backup_of != -1)
				{
					arg->msg.checkpoint_id = cp_template->cp_id;
					arg->msg.checkpoint_instance = current_instance;
					_STARPU_MALLOC(cpy_ptr, item->count);
					starpu_variable_data_register(&arg->handle, STARPU_MAIN_RAM, (uintptr_t)cpy_ptr, item->count);
					arg->rank = item->backup_of;
					_STARPU_MPI_DEBUG(0, "Submit CP: receiving external data tag:%ld, from :%d\n", arg->tag, arg->rank);
					starpu_mpi_irecv_detached(arg->handle, arg->rank, arg->tag, MPI_COMM_WORLD,
											  &_recv_cp_external_data_cb, (void*)arg);
					// The callback needs to store the received data and post ack send
				}
				break;
			case STARPU_R:
				handle = (starpu_data_handle_t)item->ptr;
				mpi_data = _starpu_mpi_data_get(handle);
				if (starpu_mpi_data_get_rank(handle)==_my_rank)
				{
					if (!mpi_data->modified)
					{
						_starpu_mpi_checkpoint_tracker_update(cp_template, cp_template->cp_id, cp_template->checkpoint_domain, current_instance);
						//TODO: check if the data are all acknowledged
						_STARPU_MPI_DEBUG(0, "Submit CP: skip send starPU data to %d (tag %d)\n", item->backupped_by, (int)starpu_mpi_data_get_tag(handle));
						_STARPU_MPI_FT_STATS_SEND_CACHED_CP_DATA(starpu_data_get_size(handle));
						break; // We don't want to CP a data that is still at initial state.
					}
					_STARPU_MPI_DEBUG(0, "Submit CP: sending starPU data to %d (tag %d)\n", item->backupped_by, (int)starpu_mpi_data_get_tag(handle));
					_STARPU_MALLOC(arg, sizeof(struct _starpu_mpi_cp_ack_arg_cb));
					arg->rank = item->backupped_by;
					arg->handle = handle;
					arg->tag = starpu_mpi_data_get_tag(handle);
					arg->type = STARPU_R;
					arg->count = item->count;
					arg->checkpoint_instance_hint = current_instance;
					_starpu_mpi_isend_cache_aware(handle, item->backupped_by, starpu_mpi_data_get_tag(handle), MPI_COMM_WORLD, 1, 0, prio,
					                              &_send_cp_internal_data_cb, (void*)arg, 1, &arg->cache_flag);
					// the callbacks need to post ack recv. The cache one needs to release the handle.
					_send_internal_data_stats(arg);
				}
				else if (item->backup_of == starpu_mpi_data_get_rank(handle))
				{
					if (!mpi_data->modified)
					{
						_STARPU_MPI_DEBUG(0, "Submit CP: skip recv starPU data to %d (tag %d)\n", item->backupped_by, (int)starpu_mpi_data_get_tag(handle));
						_STARPU_MPI_FT_STATS_RECV_CACHED_CP_DATA(starpu_data_get_size(handle));
						break; // We don't want to CP a data that is still at initial state.
					}
					_STARPU_MPI_DEBUG(0, "Submit CP: receiving starPU data from %d (tag %d)\n", starpu_mpi_data_get_rank(handle), (int)starpu_mpi_data_get_tag(handle));
					_STARPU_MALLOC(arg, sizeof(struct _starpu_mpi_cp_ack_arg_cb));
					arg->rank = item->backup_of;
					arg->handle = handle;
					arg->tag = starpu_mpi_data_get_tag(handle);
					arg->type = STARPU_R;
					arg->count = item->count;
					arg->msg.checkpoint_id = cp_template->cp_id;
					arg->msg.checkpoint_instance = current_instance;
					_starpu_mpi_irecv_cache_aware(handle, starpu_mpi_data_get_rank(handle), starpu_mpi_data_get_tag(handle), MPI_COMM_WORLD, 1, 0,
								      NULL, NULL, 1, 0, 1, &arg->cache_flag);
					// The callback needs to do nothing. The cached one must release the handle.
					//  _recv_internal_data_stats(arg);  // Now done in data_cache_set
					starpu_data_dup_ro(&arg->copy_handle, arg->handle, 1);
					starpu_data_acquire_cb(arg->copy_handle, STARPU_R, _recv_internal_dup_ro_cb, arg);
					// The callback need to store the data and post ack send.
				}
				break;
		}

		item = _starpu_mpi_checkpoint_template_get_next_data(cp_template, item);
	}

	return 0;
}

//
///**
// * receives param of type starpu_mpi_checkpoint_template_t
// * @param args
// * @return
// */
//void _starpu_mpi_checkpoint_ack_send_cb(void* args)
//{
//	starpu_mpi_checkpoint_template_t cp_template = (starpu_mpi_checkpoint_template_t) args;
//	starpu_pthread_mutex_lock(&cp_template->mutex);
//	cp_template->remaining_ack_awaited--;
//	starpu_pthread_mutex_unlock(&cp_template->mutex);
//}


//
//void _starpu_checkpoint_cached_data_send_copy_and_ack(void* _arg)
//{
//	struct _starpu_mpi_cp_ack_arg_cb* arg = (struct _starpu_mpi_cp_ack_arg_cb*) _arg;
//	starpu_data_register_same(&arg->copy_handle, arg->handle);
//	starpu_data_cpy_priority(arg->copy_handle, arg->handle, 1, _starpu_mpi_push_cp_ack_recv_cb, _arg, STARPU_MAX_PRIO);
//	starpu_data_release(arg->handle);
//}
//
//void _starpu_checkpoint_data_send_copy_and_ack(void* _args)
//{
//	struct _starpu_mpi_cp_ack_arg_cb* arg = (struct _starpu_mpi_cp_ack_arg_cb*) _args;
//	starpu_data_register_same(&arg->copy_handle, arg->handle);
//	starpu_data_cpy_priority(arg->copy_handle, arg->handle, 1, _starpu_mpi_push_cp_ack_recv_cb, _args, STARPU_MAX_PRIO);
//}
//
//void _starpu_mpi_treat_cache_ack_no_lock_cb(void* _args)
//{
//	starpu_mpi_checkpoint_template_t cp_template = (starpu_mpi_checkpoint_template_t)_args;
//	cp_template->remaining_ack_awaited--;
//}

