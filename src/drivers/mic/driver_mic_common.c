/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2012-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Thibaut Lambert
 * Copyright (C) 2015       Mathieu Lirzin
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


#include <starpu.h>
#include <drivers/mp_common/mp_common.h>
#include <drivers/mic/driver_mic_common.h>
#include <drivers/mic/driver_mic_source.h>

void _starpu_mic_common_report_scif_error(const char *func, const char *file, const int line, const int status)
{
	const char *errormsg = strerror(status);
	_STARPU_ERROR("Common: oops in %s (%s:%d)... %d: %s \n", func, file, line, status, errormsg);
}

/* Handles the error so the caller (which must be generic) doesn't have to
 * care about it.
 */

void _starpu_mic_common_send(const struct _starpu_mp_node *node, void *msg, int len)
{
  if ((scif_send(node->mp_connection.mic_endpoint, msg, len, SCIF_SEND_BLOCK)) < 0)
		STARPU_MP_COMMON_REPORT_ERROR(node, errno);
}


/* Teel is the mic endpoint is ready
 * return 1 if a message has been receive, 0 if no message has been receive
 */
int _starpu_mic_common_recv_is_ready(const struct _starpu_mp_node *mp_node)
{
  struct scif_pollepd pollepd;
  pollepd.epd = mp_node->mp_connection.mic_endpoint;
  pollepd.events = SCIF_POLLIN;
  pollepd.revents = 0;
  return  scif_poll(&pollepd,1,0);

}


/* Handles the error so the caller (which must be generic) doesn't have to
 * care about it.
 */

void _starpu_mic_common_recv(const struct _starpu_mp_node *node, void *msg, int len)
{
	if ((scif_recv(node->mp_connection.mic_endpoint, msg, len, SCIF_RECV_BLOCK)) < 0)
		STARPU_MP_COMMON_REPORT_ERROR(node, errno);
}

/* Handles the error so the caller (which must be generic) doesn't have to
 * care about it.
 */
void _starpu_mic_common_dt_send(const struct _starpu_mp_node *mp_node, void *msg, int len, void * event)
{
	if ((scif_send(mp_node->host_sink_dt_connection.mic_endpoint, msg, len, SCIF_SEND_BLOCK)) < 0)
		STARPU_MP_COMMON_REPORT_ERROR(mp_node, errno);
}

/* Handles the error so the caller (which must be generic) doesn't have to
 * care about it.
 */
void _starpu_mic_common_dt_recv(const struct _starpu_mp_node *mp_node, void *msg, int len, void * event)
{
	if ((scif_recv(mp_node->host_sink_dt_connection.mic_endpoint, msg, len, SCIF_SEND_BLOCK)) < 0)
		STARPU_MP_COMMON_REPORT_ERROR(mp_node, errno);
}

void _starpu_mic_common_connect(scif_epd_t *endpoint, uint16_t remote_node, COIPROCESS process,
				uint16_t local_port_number, uint16_t remote_port_number)
{
	/* Endpoint only useful for the initialization of the connection */
	struct scif_portID portID;

	portID.node = remote_node;
	portID.port = remote_port_number;

	if ((*endpoint = scif_open()) < 0)
		STARPU_MIC_COMMON_REPORT_SCIF_ERROR(errno);

	if ((scif_bind(*endpoint, local_port_number)) < 0)
		STARPU_MIC_COMMON_REPORT_SCIF_ERROR(errno);

	_STARPU_DEBUG("Connecting to MIC %d on %d:%d...\n", remote_node, local_port_number, remote_port_number);
	while (scif_connect(*endpoint, &portID) == -1)
	{
		if (process)
		{
			const char *main_name = "starpu_init";
			COIFUNCTION func;
			COIRESULT res;
			/* Check whether it's still alive */
			res = COIProcessGetFunctionHandles(process, 1, &main_name, &func);
			STARPU_ASSERT_MSG(res != COI_PROCESS_DIED, "process died on MIC %d", remote_node-1);
			STARPU_ASSERT_MSG(res != COI_DOES_NOT_EXIST, "MIC program does not expose the 'starpu_init' function, please link it with -rdynamic or -export-dynamic");
			if (res != COI_SUCCESS)
				STARPU_MIC_SRC_REPORT_COI_ERROR(res);
		}
		if (errno != ECONNREFUSED)
			STARPU_MIC_COMMON_REPORT_SCIF_ERROR(errno);
	}
	_STARPU_DEBUG("done\n");
}

/* Wait and accept the connection from the wanted device on the port PORT_NUMBER
 * and then initialize the connection, the resutling endpoint is stored in ENDPOINT */
void _starpu_mic_common_accept(scif_epd_t *endpoint, uint16_t port_number)
{
	/* Unused variables, only useful to make scif_accept don't cause
	 * a seg fault when trying to access PEER parameter */
	struct scif_portID portID;

	/* Endpoint only useful for the initialization of the connection */
	int init_epd;

	if ((init_epd = scif_open()) < 0)
		STARPU_MIC_COMMON_REPORT_SCIF_ERROR(errno);

	if ((scif_bind(init_epd, port_number)) < 0)
		STARPU_MIC_COMMON_REPORT_SCIF_ERROR(errno);

	/* We fix the maximum number of request to 1 as we
	 * only need one connection, more would be an error */
	if ((scif_listen(init_epd, 1)) < 0)
		STARPU_MIC_COMMON_REPORT_SCIF_ERROR(errno);

	_STARPU_DEBUG("MIC accepting connection on %u...\n", port_number);
	if ((scif_accept(init_epd, &portID, endpoint, SCIF_ACCEPT_SYNC)) < 0)
		STARPU_MIC_COMMON_REPORT_SCIF_ERROR(errno);
	_STARPU_DEBUG("done : %d\n", init_epd);

	scif_close(init_epd);
}
