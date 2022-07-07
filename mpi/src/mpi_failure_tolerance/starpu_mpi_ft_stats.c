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

#include <mpi_failure_tolerance/starpu_mpi_ft_stats.h>

starpu_pthread_mutex_t _ft_stats_mutex;

int cp_data_msgs_sent_count;
size_t cp_data_msgs_sent_total_size;
int cp_data_msgs_received_count;
size_t cp_data_msgs_received_total_size;

int cp_data_msgs_sent_cached_count;
size_t cp_data_msgs_sent_cached_total_size;
int cp_data_msgs_received_cached_count;
size_t cp_data_msgs_received_cached_total_size;
int cp_data_msgs_received_cp_cached_count;
size_t cp_data_msgs_received_cp_cached_total_size;

int ft_service_msgs_sent_count;
size_t ft_service_msgs_sent_total_size;
int ft_service_msgs_received_count;
size_t ft_service_msgs_received_total_size;

struct size_sample_list cp_data_in_memory_list; //over time
size_t cp_data_in_memory_size_max_at_t;
size_t cp_data_in_memory_size_total;

