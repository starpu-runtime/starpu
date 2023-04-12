/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2023-  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#ifndef __DRIVER_TCPIP_COMMON_H__
#define __DRIVER_TCPIP_COMMON_H__

/** @file */

#include <drivers/mp_common/mp_common.h>
#include <drivers/tcpip/driver_tcpip_source.h>

#pragma GCC visibility push(hidden)

#ifdef STARPU_USE_TCPIP_MASTER_SLAVE

extern int _starpu_tcpip_common_multiple_thread;

struct _starpu_tcpip_socket
{
	/* socket used for synchronous communications*/
	int sync_sock;
	/* socket used for asynchronous communications*/
	int async_sock;
	/* socket used for notification communications*/
	int notif_sock;
	/* a flag to detect whether the socket can be used for MSG_ZEROCOPY */
	int zerocopy;
	/* how many times is this message split up to send */
	unsigned nbsend;
	unsigned nback;
};

extern struct _starpu_tcpip_socket *tcpip_sock;

int _starpu_tcpip_mp_has_local();

int _starpu_tcpip_common_mp_init();
void _starpu_tcpip_common_mp_deinit();

int _starpu_tcpip_common_is_src_node();
int _starpu_tcpip_common_get_src_node();
int _starpu_tcpip_common_is_mp_initialized();
int _starpu_tcpip_common_recv_is_ready(const struct _starpu_mp_node *mp_node);
int _starpu_tcpip_common_notif_recv_is_ready(const struct _starpu_mp_node *mp_node);
int _starpu_tcpip_common_notif_send_is_ready(const struct _starpu_mp_node *mp_node);
void _starpu_tcpip_common_wait(struct _starpu_mp_node *mp_node);
void _starpu_tcpip_common_signal(const struct _starpu_mp_node *mp_node);

void _starpu_tcpip_common_mp_initialize_src_sink(struct _starpu_mp_node *node);

void _starpu_tcpip_common_send(const struct _starpu_mp_node *node, void *msg, int len, void * event);
void _starpu_tcpip_common_recv(const struct _starpu_mp_node *node, void *msg, int len, void * event);

void _starpu_tcpip_common_mp_send(const struct _starpu_mp_node *node, void *msg, int len);
void _starpu_tcpip_common_mp_recv(const struct _starpu_mp_node *node, void *msg, int len);

void _starpu_tcpip_common_nt_send(const struct _starpu_mp_node *node, void *msg, int len);
void _starpu_tcpip_common_nt_recv(const struct _starpu_mp_node *node, void *msg, int len);

void _starpu_tcpip_common_recv_from_device(const struct _starpu_mp_node *node, int devid, void *msg, int len, void * event);
void _starpu_tcpip_common_send_to_device(const struct _starpu_mp_node *node, int devid, void *msg, int len, void * event);

unsigned int _starpu_tcpip_common_test_event(struct _starpu_async_channel * event);
void _starpu_tcpip_common_wait_request_completion(struct _starpu_async_channel * event);

void _starpu_tcpip_common_barrier(void);

void _starpu_tcpip_common_measure_bandwidth_latency(double bandwidth_dtod[STARPU_MAXTCPIPDEVS][STARPU_MAXTCPIPDEVS], double latency_dtod[STARPU_MAXTCPIPDEVS][STARPU_MAXTCPIPDEVS]);

#endif  /* STARPU_USE_TCPIP_MASTER_SLAVE */

#pragma GCC visibility pop

#endif	/* __DRIVER_TCPIP_COMMON_H__ */
