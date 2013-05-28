/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012  Inria
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


#ifndef __DRIVER_MIC_COMMON_H__
#define __DRIVER_MIC_COMMON_H__


#include <common/config.h>


#ifdef STARPU_USE_MIC

#define STARPU_TO_MIC_ID(id) ((id) + 1)

#define STARPU_MIC_PORTS_BEGIN 1099

#define STARPU_MIC_SOURCE_PORT_NUMBER STARPU_MIC_PORTS_BEGIN
#define STARPU_MIC_SINK_PORT_NUMBER(id) ((id) + STARPU_MIC_PORTS_BEGIN + 1)

#define STARPU_MIC_SOURCE_DT_PORT_NUMBER (STARPU_MAXMICDEVS + STARPU_MIC_PORTS_BEGIN + 1)
#define STARPU_MIC_SINK_DT_PORT_NUMBER(id) ((id) + STARPU_MAXMICDEVS + STARPU_MIC_PORTS_BEGIN + 2)

#define STARPU_MIC_SINK_SINK_DT_PORT_NUMBER(me, peer_id) \
((me) * STARPU_MAXMICDEVS + (peer_id) +  2 * STARPU_MAXMICDEVS + STARPU_MIC_PORTS_BEGIN + 2)

#define STARPU_MIC_PAGE_SIZE 0x1000
#define STARPU_MIC_GET_PAGE_SIZE_MULTIPLE(size) \
(((size) % STARPU_MIC_PAGE_SIZE == 0) ? (size) : (((size) / STARPU_MIC_PAGE_SIZE + 1) * STARPU_MIC_PAGE_SIZE))

#define STARPU_MIC_COMMON_REPORT_SCIF_ERROR(status) \
	_starpu_mic_common_report_scif_error(__starpu_func__, __FILE__, __LINE__, status)

struct _starpu_mic_free_command
{
	void *addr;
	size_t size;
};

void _starpu_mic_common_report_scif_error(const char *func, const char *file, int line, const int status);

void _starpu_mic_common_send(const struct _starpu_mp_node *node, void *msg, int len);

void _starpu_mic_common_recv(const struct _starpu_mp_node *node, void *msg, int len);

void _starpu_mic_common_dt_send(const struct _starpu_mp_node *node, void *msg, int len);

void _starpu_mic_common_dt_recv(const struct _starpu_mp_node *node, void *msg, int len);

void _starpu_mic_common_connect(scif_epd_t *endpoint, uint16_t remote_node, 
				uint16_t local_port_number, uint16_t remote_port_number);
void _starpu_mic_common_accept(scif_epd_t *endpoint, uint16_t port_number);

#endif /* STARPU_USE_MIC */

#endif /* __DRIVER_MIC_COMMON_H__ */
