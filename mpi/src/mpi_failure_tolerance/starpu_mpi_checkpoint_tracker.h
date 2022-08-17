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

#ifndef FT_STARPU_STARPU_MPI_CHECKPOINT_TRACKER_H
#define FT_STARPU_STARPU_MPI_CHECKPOINT_TRACKER_H

#ifdef __cplusplus
extern "C"
{
#endif

struct _starpu_mpi_checkpoint_tracker
{
	int                              cp_id;
	int                              cp_inst;
	int                              cp_domain;
	starpu_mpi_checkpoint_template_t cp_template;
	int                              ack_msg_count;
	int                              first_msg_sent_flag;
	int                              old:1;
	int                              valid: 1;
};

int _starpu_mpi_checkpoint_tracker_init();
int _starpu_mpi_checkpoint_tracker_shutdown();
struct _starpu_mpi_checkpoint_tracker* _starpu_mpi_checkpoint_template_get_tracking_inst_by_id_inst(int cp_domain, int cp_inst);
struct _starpu_mpi_checkpoint_tracker* _starpu_mpi_checkpoint_template_create_instance_tracker(starpu_mpi_checkpoint_template_t cp_template, int cp_id, int cp_domain, int cp_inst);
struct _starpu_mpi_checkpoint_tracker* _starpu_mpi_checkpoint_tracker_update(starpu_mpi_checkpoint_template_t cp_template, int cp_id, int cp_domain, int cp_instance);
int _starpu_mpi_checkpoint_check_tracker(struct _starpu_mpi_checkpoint_tracker* tracker);
struct _starpu_mpi_checkpoint_tracker* _starpu_mpi_checkpoint_tracker_validate_instance(struct _starpu_mpi_checkpoint_tracker* tracker);
struct _starpu_mpi_checkpoint_tracker* _starpu_mpi_checkpoint_tracker_get_last_valid_tracker(int domain);

#ifdef __cplusplus
}
#endif

#endif //FT_STARPU_STARPU_MPI_CHECKPOINT_TRACKER_H
