/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <sys/socket.h>
#include <netinet/in.h>
#include <sys/un.h>
#ifdef HAVE_UNISTD_H
#include <unistd.h>
#endif
#include <error.h>
#include <errno.h>
#include <linux/errqueue.h>
#include <sys/types.h>
#include <netdb.h>
#include <arpa/inet.h>
#include <poll.h>
#include <pthread.h>
#include <signal.h>
#include <core/workers.h>
#include <core/perfmodel/perfmodel.h>
#include <drivers/mp_common/source_common.h>
#include <drivers/tcpip/driver_tcpip_common.h>
#include <common/starpu_spinlock.h>
#include <common/uthash.h>
#include <drivers/tcpip/driver_tcpip_common_func.h>

#define NITER 32
#define SIZE_BANDWIDTH (1024*1024)

#define _SELECT_DEBUG 0
#if _SELECT_DEBUG
#  define _SELECT_PRINT(...) printf(__VA_ARGS__)
#else
#  define _SELECT_PRINT(...)
#endif

#define _ZC_DEBUG 0
#if _ZC_DEBUG
#  define _ZC_PRINT(...) printf(__VA_ARGS__)
#else
#  define _ZC_PRINT(...)
#endif

typedef starpu_ssize_t(*what_t)(int fd, void *buf, size_t count);

static int tcpip_initialized = 0;
static int extern_initialized = 0;
//static int src_node_id = 0;
static int nb_sink;
static char* host_port;
static int index_sink = 0;

int _starpu_tcpip_common_multiple_thread;

static int is_running;

static struct _starpu_spinlock ListLock;

static starpu_pthread_t thread_pending;
static int thread_pipe[2];

static pthread_t master_thread;

struct _starpu_tcpip_socket *tcpip_sock;

/* a flag to note whether the socket is local socket*/
static int *local_flag;

int _starpu_tcpip_mp_has_local()
{
	for (int i=1; i<=nb_sink; i++)
	{
		if(local_flag[i] == 1)
			return 1;
	}

	return 0;
}

MULTILIST_CREATE_TYPE(_starpu_tcpip_ms_request, event); /*_starpu_tcpip_ms_request_multilist_event*/
MULTILIST_CREATE_TYPE(_starpu_tcpip_ms_request, thread); /*_starpu_tcpip_ms_request_multilist_thread*/
MULTILIST_CREATE_TYPE(_starpu_tcpip_ms_request, pending); /*_starpu_tcpip_ms_request_multilist_pending*/

struct _starpu_tcpip_ms_request
{
	/*member of list of event*/
	struct _starpu_tcpip_ms_request_multilist_event event;
	/*member of list of thread for async send/receive*/
	struct _starpu_tcpip_ms_request_multilist_thread thread;
	/*member of list of pending for except in select*/
	struct _starpu_tcpip_ms_request_multilist_pending pending;
	/*the struct of remote socket to send/receive message*/
	struct _starpu_tcpip_socket *remote_sock;
	/*the message to send/receive*/
	char* buf;
	/*the length of message*/
	int len;
	/*a flag to detect wether the operation is completed*/
	int flag_completed;
	/*a semaphore to detect wether the request is completed*/
	starpu_sem_t sem_wait_request;
	/*a flag to detect send or receive*/
	int is_sender;
	/*the length of message that has been sent/wrote*/
	int offset;
	/*active the flag MSG_ZEROCOPY*/
	int zerocopy;
	/*record the count at the end of send*/
	uint32_t send_end;
};

MULTILIST_CREATE_INLINES(struct _starpu_tcpip_ms_request, _starpu_tcpip_ms_request, event);
MULTILIST_CREATE_INLINES(struct _starpu_tcpip_ms_request, _starpu_tcpip_ms_request, thread);
MULTILIST_CREATE_INLINES(struct _starpu_tcpip_ms_request, _starpu_tcpip_ms_request, pending);

static struct _starpu_tcpip_ms_request_multilist_thread thread_list;

struct _starpu_tcpip_ms_async_event
{
	int is_sender;
	struct _starpu_tcpip_ms_request_multilist_event *requests;
};

static inline struct _starpu_tcpip_ms_async_event *_starpu_tcpip_ms_async_event(union _starpu_async_channel_event *_event)
{
	struct _starpu_tcpip_ms_async_event *event;
	STARPU_STATIC_ASSERT(sizeof(*event) <= sizeof(*_event));
	event = (void *) _event;
	return event;
}

/*hash table struct*/
struct _starpu_tcpip_req_pending
{
	int remote_sock;
	struct _starpu_tcpip_ms_request_multilist_thread send_list;
	struct _starpu_tcpip_ms_request_multilist_thread recv_list;
	struct _starpu_tcpip_ms_request_multilist_pending pending_list;
	UT_hash_handle hh;
};

//function thread
static void * _starpu_tcpip_thread_pending()
{
	fd_set reads;
	fd_set writes;
	int fdmax=0;

	struct _starpu_tcpip_req_pending *pending_tables = NULL;
	struct _starpu_tcpip_req_pending *table, *tmp;

	FD_ZERO(&reads);
	FD_ZERO(&writes);

	FD_SET(thread_pipe[0], &reads);
	fd_set reads2;
	fd_set writes2;

	fdmax = thread_pipe[0];

	while(is_running)
	{
		_SELECT_PRINT("in while\n");
		reads2 = reads;
		writes2 = writes;

		int ret;
		ret=select(fdmax+1, &reads2, &writes2, NULL, NULL);
		STARPU_ASSERT(ret>=0);

		if(FD_ISSET(thread_pipe[0], &reads2))
		{
			char buf[16];
			int n=read(thread_pipe[0], buf, sizeof(buf));
			STARPU_ASSERT(n>=0);
			if(!is_running)
				break;

			int i;
			for(i=0; i<n; i++)
			{
				_SELECT_PRINT("pop push loop %d\n", i);
				_starpu_spin_lock(&ListLock);
				STARPU_ASSERT(!_starpu_tcpip_ms_request_multilist_empty_thread(&thread_list));
				struct _starpu_tcpip_ms_request * req_thread = _starpu_tcpip_ms_request_multilist_pop_front_thread(&thread_list);
				_starpu_spin_unlock(&ListLock);

				int remote_sock = req_thread->remote_sock->async_sock;
				int is_sender = req_thread->is_sender;

				HASH_FIND_INT(pending_tables, &remote_sock, table);
				if(table == NULL)
				{
					_STARPU_MALLOC(table, sizeof(*table));
					table->remote_sock = remote_sock;
					_starpu_tcpip_ms_request_multilist_head_init_thread(&table->send_list);
					_starpu_tcpip_ms_request_multilist_head_init_thread(&table->recv_list);
					_starpu_tcpip_ms_request_multilist_head_init_pending(&table->pending_list);
					HASH_ADD_INT(pending_tables, remote_sock, table);

				}
				if(is_sender)
				{
					_starpu_tcpip_ms_request_multilist_push_back_thread(&table->send_list, req_thread);
					FD_SET(remote_sock, &writes);
				}
				else
				{
					_starpu_tcpip_ms_request_multilist_push_back_thread(&table->recv_list, req_thread);
					FD_SET(remote_sock, &reads);
				}

				if(remote_sock > fdmax)
					fdmax=remote_sock;
			}

		}

		HASH_ITER(hh, pending_tables, table, tmp)
		{
			int remote_sock = table->remote_sock;
			_SELECT_PRINT("remote_sock in loop is %d\n", remote_sock);

			void socket_action(what_t what, const char * whatstr, struct _starpu_tcpip_ms_request_multilist_thread *list, fd_set * fdset)
			{
				struct _starpu_tcpip_ms_request * req = _starpu_tcpip_ms_request_multilist_begin_thread(list);
				char* msg = req->buf;
				int len = req->len;

				int res = 0;
				res = what(remote_sock, msg+req->offset, len-req->offset);
				_SELECT_PRINT("%s res is %d\n", whatstr, res);
				STARPU_ASSERT_MSG(res > 0, "TCP/IP Master/Slave cannot %s a msg asynchronous with a size of %d Bytes!, the result of %s is %d, the error is %s ", whatstr, len, whatstr, res, strerror(errno));
				req->offset+=res;

				_SELECT_PRINT("offset after %s is %d\n", whatstr, req->offset);

				if(req->offset == len)
				{
					_starpu_tcpip_ms_request_multilist_erase_thread(list, req);

					if(_starpu_tcpip_ms_request_multilist_empty_thread(list))
						FD_CLR(remote_sock, fdset);

					req->flag_completed = 1;
					starpu_sem_post(&req->sem_wait_request);

					/*send the signal that message is ready */
					struct _starpu_mp_node *node = NULL;
					_starpu_tcpip_common_signal(node);
				}
			}

			if(FD_ISSET(remote_sock, &writes2))
			{
#ifdef SO_ZEROCOPY
				struct pollfd pfd;
				pfd.fd = remote_sock;
				pfd.events = POLLERR|POLLOUT;
				pfd.revents = 0;
				if(poll(&pfd, 1, -1) <= 0)
					error(1, errno, "poll");

				if(pfd.revents & POLLERR)
				{
					struct _starpu_tcpip_ms_request * req_pending = _starpu_tcpip_ms_request_multilist_begin_pending(&table->pending_list);
					_ZC_PRINT("nbsend is %d\n", req_pending->remote_sock->nbsend);
					struct sock_extended_err *serr;
					struct msghdr mg = {};
					struct cmsghdr *cm;
					uint32_t hi, lo;
					char control[100];

					mg.msg_control = control;
					mg.msg_controllen = sizeof(control);

					_ZC_PRINT("before recvmsg\n");
					int r = recvmsg(remote_sock, &mg, MSG_ERRQUEUE);
					// if (r == -1 && errno == EAGAIN)
					//	   continue;
					if (r == -1)
						error(1, errno, "recvmsg notification");
					if (mg.msg_flags & MSG_CTRUNC)
						error(1, errno, "recvmsg notification: truncated");

					cm = CMSG_FIRSTHDR(&mg);
					if (!cm)
						error(1, 0, "cmsg: no cmsg");

					serr = (void *) CMSG_DATA(cm);

					if (serr->ee_origin != SO_EE_ORIGIN_ZEROCOPY)
						error(1, 0, "serr: wrong origin: %u", serr->ee_origin);
					if (serr->ee_errno != 0)
						error(1, 0, "serr: wrong error code: %u", serr->ee_errno);

					if (serr->ee_code != SO_EE_CODE_ZEROCOPY_COPIED)
						req_pending->zerocopy = 0;

					hi = serr->ee_data;
					lo = serr->ee_info;

					_ZC_PRINT("h=%u l=%u\n", hi, lo);

					STARPU_ASSERT(lo == req_pending->remote_sock->nback);
					STARPU_ASSERT(hi < req_pending->remote_sock->nbsend);

					req_pending->remote_sock->nback = hi+1;

					_ZC_PRINT("send end is %d\n", req_pending->send_end);
					while(!_starpu_tcpip_ms_request_multilist_empty_pending(&table->pending_list))
					{
						struct _starpu_tcpip_ms_request * req_tmp = _starpu_tcpip_ms_request_multilist_begin_pending(&table->pending_list);

						if(hi+1 >= req_tmp->send_end)
						{
							_starpu_tcpip_ms_request_multilist_erase_pending(&table->pending_list, req_tmp);

							if(_starpu_tcpip_ms_request_multilist_empty_thread(&table->send_list)&&_starpu_tcpip_ms_request_multilist_empty_pending(&table->pending_list))
								FD_CLR(remote_sock, &writes);

							req_tmp->flag_completed = 1;
							starpu_sem_post(&req_tmp->sem_wait_request);

							/*send the signal that message is ready*/
							struct _starpu_mp_node *node = NULL;
							_starpu_tcpip_common_signal(node);
						}
						else
							break;
					}

				}
				else
				{
					if(!(_starpu_tcpip_ms_request_multilist_empty_thread(&table->send_list)))
					{
						struct _starpu_tcpip_ms_request * req = _starpu_tcpip_ms_request_multilist_begin_thread(&table->send_list);
						char* msg = req->buf;
						int len = req->len;

						if(req->remote_sock->zerocopy)
						{
							_ZC_PRINT("msg len is %d\n", len);
							_ZC_PRINT("offset before send is %d\n", req->offset);

							if(req->offset == 0)
							{
								_starpu_tcpip_ms_request_multilist_push_back_pending(&table->pending_list, req);
							}

							int res = send(remote_sock, msg+req->offset, len-req->offset, MSG_ZEROCOPY);
							_ZC_PRINT("send return %d\n", res);
							STARPU_ASSERT_MSG(res > 0, "TCP/IP Master/Slave cannot send a msg asynchronous with a size of %d Bytes!, the result of send is %d, the error is %s ", len, res, strerror(errno));

							req->remote_sock->nbsend++;
							req->offset+=res;

							_ZC_PRINT("offset after send is %d\n", req->offset);

							if(req->offset == len)
							{
								req->send_end = req->remote_sock->nbsend;
								_ZC_PRINT("send end after send is %d\n", req->send_end);
								_starpu_tcpip_ms_request_multilist_erase_thread(&table->send_list, req);

								//if(_starpu_tcpip_ms_request_multilist_empty_thread(&table->send_list))
									//we need this to check whether the msg are all sent, we would have to remove POLLOUT from poll.events
									//FD_CLR(remote_sock, &writes);
							}
						}
						else
#endif
						{
							socket_action((what_t)write, "write", &table->send_list, &writes);
						}
#ifdef SO_ZEROCOPY
					}
				}
#endif
			}

			if(FD_ISSET(remote_sock, &reads2))
			{
				socket_action(read, "read", &table->recv_list, &reads);
			}
			/*if the recv/send_list is empty, delete and free hash table*/
			if(_starpu_tcpip_ms_request_multilist_empty_thread(&table->send_list)&&_starpu_tcpip_ms_request_multilist_empty_thread(&table->recv_list)&&_starpu_tcpip_ms_request_multilist_empty_pending(&table->pending_list))
			{
				HASH_DEL(pending_tables, table);
				free(table);
			}

		}

	}
	/*all hash tables should be deleted*/
	STARPU_ASSERT(pending_tables == NULL);

	return 0;
}

static void handler(int num STARPU_ATTRIBUTE_UNUSED){}

int _starpu_tcpip_common_mp_init()
{
	//Here we supposed the programmer called two times starpu_init.
	if (tcpip_initialized)
		return -ENODEV;

	/*get the slave number*/
	nb_sink = starpu_getenv_number("STARPU_TCPIP_MS_SLAVES");
	//_TCPIP_PRINT("the slave number is %d\n", nb_sink);

	if (nb_sink <= 0)
		/* No slave */
		return 0;

	tcpip_initialized = 1;

	_starpu_tcpip_common_multiple_thread = starpu_getenv_number_default("STARPU_TCPIP_MS_MULTIPLE_THREAD", 0);

	master_thread = pthread_self();
	signal(SIGUSR1, handler);

	/*initialize the pipe*/
	int r=pipe(thread_pipe);
	STARPU_ASSERT(r==0);

	_starpu_spin_init(&ListLock);
	/*initialize the thread*/
	_starpu_tcpip_ms_request_multilist_head_init_thread(&thread_list);

	STARPU_HG_DISABLE_CHECKING(is_running);
	is_running = 1;
	STARPU_PTHREAD_CREATE(&thread_pending, NULL, _starpu_tcpip_thread_pending, NULL);

	/*get host info*/
	host_port = starpu_getenv("STARPU_TCPIP_MS_MASTER");

	_STARPU_CALLOC(tcpip_sock, nb_sink + 1, sizeof(struct _starpu_tcpip_socket));
	_STARPU_MALLOC(local_flag, (nb_sink + 1)*sizeof(int));

	struct sockaddr_in* sink_addr_list;
	_STARPU_MALLOC(sink_addr_list, (nb_sink + 1)*sizeof(struct sockaddr_in));

#if _TCPIP_DEBUG
	char clnt_ip[20];
#endif
	/*master part*/
	if(!host_port)
	{
		int source_sock_init = 0;
		int local_sock = 0;
		struct sockaddr_un name;
		struct sockaddr_in source_addr_init;
		socklen_t source_addr_init_size = sizeof(source_addr_init);

		unsigned short port = starpu_getenv_number_default("STARPU_TCPIP_MS_PORT", 1234);

		int init_res = master_init(1, &source_sock_init, &local_sock, &source_addr_init, &source_addr_init_size, &name, htonl(INADDR_ANY), htons(port), 2*nb_sink);
		if(init_res != 0)
			return -1;

		_TCPIP_PRINT("source_sock_init is %d\n", source_sock_init);
		_TCPIP_PRINT("local_sock is %d\n", local_sock);
		tcpip_sock[0].sync_sock = -1;
		tcpip_sock[0].async_sock = -1;
		tcpip_sock[0].notif_sock = -1;
		tcpip_sock[0].zerocopy = -1;
		/*source socket is not local socket*/
		if(local_sock == 0)
			local_flag[0] = 0;
		/*source socket is local socket*/
		else
			local_flag[0] = 1;

		int i;
		/*connect each slave, generate sync socket*/
		for (i=1; i<=nb_sink; i++)
		{
			int sink_sock;
			int local_sock_flag;
			int accept_res = master_accept(&sink_sock, source_sock_init, local_sock, NULL, &local_sock_flag);
			if(accept_res != 0)
				return -1;

			_TCPIP_PRINT("sink_sock is %d\n", sink_sock);
			tcpip_sock[i].sync_sock = sink_sock;
			local_flag[i] = local_sock_flag;
		}
		for (i=1; i<=nb_sink; i++)
		{
			/*write the id to slave*/
			int id_sink = i;
			WRITE(tcpip_sock[i].sync_sock, &id_sink, sizeof(id_sink));

			_TCPIP_PRINT("write to slave %d its index\n", id_sink);

			/*receive the slave address with the random allocated port number connect to other slaves*/
			struct sockaddr_in buf_addr;
			READ(tcpip_sock[i].sync_sock, &buf_addr, sizeof(buf_addr));

			sink_addr_list[i] = buf_addr;
			_TCPIP_PRINT("Message from slave (slave address) is , ip : %s, port : %d.\n",
			inet_ntop(AF_INET, &sink_addr_list[i].sin_addr, clnt_ip, sizeof(clnt_ip)), ntohs(sink_addr_list[i].sin_port));

		}
		/*connect each slave, generate async socket and notif socket*/
		for (i=1; i<=2*nb_sink; i++)
		{
			int sink_sock2;
			int zerocopy;
			int accept_res = master_accept(&sink_sock2, source_sock_init, local_sock, &zerocopy, NULL);
			if(accept_res != 0)
				return -1;

			int i_sink;
			/*get slave index*/
			READ(sink_sock2, &i_sink, sizeof(i_sink));

			_TCPIP_PRINT("the index received is %d, the index in loop is %d\n", i_sink, i);
			_TCPIP_PRINT("sink_sock2 is %d\n", sink_sock2);
			if(tcpip_sock[i_sink].async_sock == 0)
			{
				tcpip_sock[i_sink].async_sock = sink_sock2;
				tcpip_sock[i_sink].zerocopy = zerocopy;
			}
			else
			{
				STARPU_ASSERT(tcpip_sock[i_sink].notif_sock == 0);
				tcpip_sock[i_sink].notif_sock = sink_sock2;
			}

		}

		close(source_sock_init);
		if (starpu_getenv_number_default("STARPU_TCPIP_USE_LOCAL_SOCKET", 1) != 0)
		{
			close(local_sock);
			unlink(name.sun_path);
		}

		for(i=0; i<=nb_sink; i++)
		{
			_TCPIP_PRINT("sock_list[%d] in master part is %d\n", i, tcpip_sock[i].sync_sock);
		}
		for(i=0; i<=nb_sink; i++)
		{
			_TCPIP_PRINT("async_sock_list[%d] in master part is %d\n", i, tcpip_sock[i].async_sock);
		}
		for(i=0; i<=nb_sink; i++)
		{
			_TCPIP_PRINT("notif_sock_list[%d] in master part is %d\n", i, tcpip_sock[i].notif_sock);
		}
		/*write the address of one slave to another*/
		int j;
		for (i=1; i<=nb_sink; i++)
		{
			for(j=1; j<i; j++)
			{
				_TCPIP_PRINT("address of other slaves sent by master is, ip : %s, port : %d.\n",
				inet_ntop(AF_INET, &sink_addr_list[j].sin_addr, clnt_ip, sizeof(clnt_ip)), ntohs(sink_addr_list[j].sin_port));
				/*send address of other sinks to slave*/
				WRITE(tcpip_sock[i].sync_sock, &sink_addr_list[j], sizeof(sink_addr_list[j]));
			}
		}

		for(i=0; i<=nb_sink; i++)
		{
			_TCPIP_PRINT("local_flag[%d] in master part is %d\n", i, local_flag[i]);
		}

	}

	/*slave part*/
	else
	{
		/***************************connection between master and slave*************************/
		STARPU_ASSERT_MSG(host_port != NULL, "Slave should provide the host to connect");
		char *host;
		char *port;
		char *ret = strchr(host_port, ':');
		if(ret)
		{
			host = strndup(host_port, ret-host_port);
			port = strdup(ret+1);
		}
		else
		{
			host = strdup(host_port);
			port = starpu_getenv("STARPU_TCPIP_MS_PORT");
			if (!port)
				port = "1234";
			port = strdup(port);
		}
		int source_sock;
		struct addrinfo *res,*cur;
		struct addrinfo hints;

		memset(&hints, 0, sizeof(hints));
		hints.ai_socktype = SOCK_STREAM;

		int gaierrno = getaddrinfo(host, port, &hints, &res);
		if (gaierrno)
		{
			fprintf(stderr,"getaddrinfo: %s\n", gai_strerror(gaierrno));
			return -1;
		}

		struct sockaddr_in sink_addr;
		/*init slave*/
		for(cur = res; cur; cur = cur->ai_next)
		{
			int local_sock_flag;
			int connect_res = slave_connect(&source_sock, cur, &sink_addr, NULL, NULL, &local_sock_flag);
			if(connect_res == 1)
				continue;
			else if(connect_res < 0)
				return -1;

			_TCPIP_PRINT("source_sock is %d\n", source_sock);
			tcpip_sock[0].sync_sock = source_sock;
			local_flag[0] = local_sock_flag;

			break;
		}
		freeaddrinfo(res);
		if (!cur)
		{
			fprintf(stderr, "could not connect\n");
			return -1;
		}

		/*****************************connection between slaves********************************/

		/*get slave index in master sock_list*/
		READ(source_sock, &index_sink, sizeof(index_sink));

		tcpip_sock[index_sink].sync_sock = -1;
		tcpip_sock[index_sink].async_sock = -1;
		tcpip_sock[index_sink].notif_sock = -1;
		tcpip_sock[index_sink].zerocopy = -1;

		_TCPIP_PRINT("index_sink read from master is %d\n", index_sink);

		int sink_serv_sock = 0;
		int sink_local_sock = 0;
		struct sockaddr_un sink_name;
		struct sockaddr_in sink_serv_addr;
		socklen_t sink_serv_addr_size = sizeof(sink_serv_addr);

		int init_res = master_init(0, &sink_serv_sock, &sink_local_sock, &sink_serv_addr, &sink_serv_addr_size, &sink_name, sink_addr.sin_addr.s_addr, 0, 2*(nb_sink-index_sink));
		if(init_res != 0)
			return -1;

		_TCPIP_PRINT("sink_serv_sock is %d\n", sink_serv_sock);
		_TCPIP_PRINT("sink_local_sock is %d\n", sink_local_sock);
		/*sink serv socket is not local socket*/
		if(sink_local_sock == 0)
			local_flag[index_sink] = 0;
		/*sink serv socket is local socket*/
		else
			local_flag[index_sink] = 1;

		/*send slave address to master*/
		WRITE(source_sock, &sink_serv_addr, sink_serv_addr_size);

		/*async and notif communication*/
		int source_async_sock;
		int source_notif_sock;
		struct addrinfo *res1,*cur1;
		struct addrinfo hints1;

		memset(&hints1, 0, sizeof(hints1));
		hints1.ai_socktype = SOCK_STREAM;

		int gaierrno1 = getaddrinfo(host, port, &hints1, &res1);
		if (gaierrno1)
		{
			fprintf(stderr,"getaddrinfo: %s\n", gai_strerror(gaierrno1));
			return -1;
		}

		for(cur1 = res1; cur1; cur1 = cur1->ai_next)
		{
			/*async connect*/
			int zerocopy;
			int connect_res = slave_connect(&source_async_sock, cur1, NULL, NULL, &zerocopy, NULL);
			if(connect_res == 1)
				continue;
			else if(connect_res < 0)
				return -1;

			_TCPIP_PRINT("source_async_sock is %d\n", source_async_sock);
			tcpip_sock[0].async_sock = source_async_sock;
			tcpip_sock[0].zerocopy = zerocopy;

			/*notif connect*/
			int connect_notif_res = slave_connect(&source_notif_sock, cur1, NULL, NULL, NULL, NULL);
			if(connect_notif_res == 1)
				continue;
			else if(connect_notif_res < 0)
			{
				close(source_async_sock);
				return -1;
			}

			_TCPIP_PRINT("source_notif_sock is %d\n", source_notif_sock);
			tcpip_sock[0].notif_sock = source_notif_sock;

			break;
		}
		freeaddrinfo(res1);
		if (!cur1)
		{
			fprintf(stderr, "could not connect\n");
			return -1;
		}

		/*send slave index to master async socket*/
		WRITE(source_async_sock, &index_sink, sizeof(index_sink));

		/*send slave index to master notif socket*/
		WRITE(source_notif_sock, &index_sink, sizeof(index_sink));

		/*communication between slaves*/
		int j;
		/*the active part*/
		for (j=1; j<index_sink; j++)
		{
			struct sockaddr_in serv_addr;
			socklen_t serv_addr_size = sizeof(serv_addr);
			/*get the address of other slaves from master*/
			READ(source_sock, &serv_addr, serv_addr_size);

			_TCPIP_PRINT("address of other slave is, ip : %s, port : %d.\n",
			inet_ntop(AF_INET, &serv_addr.sin_addr, clnt_ip, sizeof(clnt_ip)), ntohs(serv_addr.sin_port));

			int serv_sock;
			int local_sock_flag;
			int connect_sync_res = slave_connect(&serv_sock, NULL, NULL, &serv_addr, NULL, &local_sock_flag);
			if(connect_sync_res != 0)
				return -1;

			_TCPIP_PRINT("index_sink in slave part is %d\n", index_sink);
			/*send sink id to another slave*/
			WRITE(serv_sock, &index_sink, sizeof(index_sink));

			tcpip_sock[j].sync_sock = serv_sock;
			local_flag[j] = local_sock_flag;

			/*async connect*/
			int serv_async_sock;
			int zerocopy;
			int connect_async_res = slave_connect(&serv_async_sock, NULL, NULL, &serv_addr, &zerocopy, NULL);
			if(connect_async_res != 0)
				return -1;

			/*send sink async id to another slave*/
			WRITE(serv_async_sock, &index_sink, sizeof(index_sink));

			tcpip_sock[j].async_sock = serv_async_sock;
			tcpip_sock[j].zerocopy = zerocopy;

			/*notif connect*/
			int serv_notif_sock;
			int connect_notif_res = slave_connect(&serv_notif_sock, NULL, NULL, &serv_addr, NULL, NULL);
			if(connect_notif_res != 0)
				return -1;

			/*send sink notif id to another slave*/
			WRITE(serv_notif_sock, &index_sink, sizeof(index_sink));

			tcpip_sock[j].notif_sock = serv_notif_sock;

			_TCPIP_PRINT("sock_list[%d] in slave part is %d\n", j, serv_sock);
			_TCPIP_PRINT("sock_async_list[%d] in slave part is %d\n", j, serv_async_sock);
			_TCPIP_PRINT("sock_notif_list[%d] in slave part is %d\n", j, serv_notif_sock);
		}

		/*the passive part*/
		for (j=index_sink+1; j<=nb_sink; j++)
		{
			/*sync accept*/
			int clnt_sock;
			int local_sock_flag;
			int accept_sync_res = master_accept(&clnt_sock, sink_serv_sock, sink_local_sock, NULL, &local_sock_flag);
			if(accept_sync_res != 0)
				return -1;

			int sink_id;
			/*get sink id*/
			READ(clnt_sock, &sink_id, sizeof(sink_id));

			//_TCPIP_PRINT("index_sink in master part is %d\n", index_sink);
			tcpip_sock[sink_id].sync_sock = clnt_sock;
			local_flag[sink_id] = local_sock_flag;

			/*async accept*/
			int clnt_async_sock;
			int zerocopy;
			int accept_async_res = master_accept(&clnt_async_sock, sink_serv_sock, sink_local_sock, &zerocopy, NULL);
			if(accept_async_res != 0)
				return -1;

			int sink_async_id;
			/*get sink async id*/
			READ(clnt_async_sock, &sink_async_id, sizeof(sink_async_id));

			tcpip_sock[sink_async_id].async_sock = clnt_async_sock;
			tcpip_sock[sink_async_id].zerocopy = zerocopy;

			/*notif accept*/
			int clnt_notif_sock;
			int accept_notif_res = master_accept(&clnt_notif_sock, sink_serv_sock, sink_local_sock, NULL, NULL);
			if(accept_notif_res != 0)
				return -1;

			int sink_notif_id;
			/*get sink notif id*/
			READ(clnt_notif_sock, &sink_notif_id, sizeof(sink_notif_id));

			tcpip_sock[sink_notif_id].notif_sock = clnt_notif_sock;

			_TCPIP_PRINT("sock_list[%d] in master part is %d\n", sink_id, clnt_sock);
			_TCPIP_PRINT("sock_async_list[%d] in master part is %d\n", sink_async_id, clnt_async_sock);
			_TCPIP_PRINT("sock_notif_list[%d] in master part is %d\n", sink_notif_id, clnt_notif_sock);
		}

		close(sink_serv_sock);
		if (starpu_getenv_number_default("STARPU_TCPIP_USE_LOCAL_SOCKET", 1) != 0)
		{
			close(sink_local_sock);
			unlink(sink_name.sun_path);
		}

		{
			int i;
			for(i=0; i<=nb_sink; i++)
			{
				_TCPIP_PRINT("sock_list[%d] is %d\n", i, tcpip_sock[i].sync_sock);
			}
			for(i=0; i<=nb_sink; i++)
			{
				_TCPIP_PRINT("async_sock_list[%d] is %d\n", i, tcpip_sock[i].async_sock);
			}
			for(i=0; i<=nb_sink; i++)
			{
				_TCPIP_PRINT("notif_sock_list[%d] is %d\n", i, tcpip_sock[i].notif_sock);
			}
			for(i=0; i<=nb_sink; i++)
			{
				_TCPIP_PRINT("local_flag[%d] is %d\n", i, local_flag[i]);
			}
		}

		setenv("STARPU_SINK", "STARPU_TCPIP_MS", 1);
		free(host);
		free(port);
	}

	free(sink_addr_list);

	return 1;
}

void _starpu_tcpip_common_mp_deinit()
{
	is_running = 0;
	char buf = 0;
	write(thread_pipe[1], &buf, 1);
	STARPU_PTHREAD_JOIN(thread_pending, NULL);
	if (!extern_initialized)
	{
		int i;
		for (i=0; i<nb_sink; i++)
		{
			if (tcpip_sock[i].sync_sock == -1)
				/* Ourself */
				continue;
			close(tcpip_sock[i].sync_sock);
			close(tcpip_sock[i].async_sock);
			close(tcpip_sock[i].notif_sock);
		}
	}
	free(tcpip_sock);
	free(local_flag);
	_starpu_spin_destroy(&ListLock);
}

int _starpu_tcpip_common_is_src_node()
{
	return index_sink == 0;
}

int _starpu_tcpip_common_get_src_node()
{
	return 0;
}

int _starpu_tcpip_common_is_mp_initialized()
{
	return tcpip_initialized;
}

/* common parts to initialize a source or a sink node */
void _starpu_tcpip_common_mp_initialize_src_sink(struct _starpu_mp_node *node)
{
	struct _starpu_machine_topology *topology = &_starpu_get_machine_config()->topology;

	int ntcpipcores = starpu_getenv_number("STARPU_NTCPIPMSTHREADS");
	if (ntcpipcores == -1)
	{
		int nhyperthreads = topology->nhwpus / topology->nhwworker[STARPU_CPU_WORKER][0];
		node->nb_cores = topology->nusedpus / nhyperthreads;
	}
	else
		node->nb_cores = ntcpipcores;
}

int _starpu_tcpip_common_recv_is_ready(const struct _starpu_mp_node *mp_node)
{
	fd_set set;
	int fd = mp_node->mp_connection.tcpip_mp_connection->sync_sock;
	int res;

	struct timeval tv =
	{
		.tv_sec = 0,
		.tv_usec = 0
	};

	FD_ZERO(&set);
	FD_SET(fd, &set);

	while((res = select(fd+1, &set, NULL, NULL, &tv)) == -1 && errno == EINTR);

	STARPU_ASSERT_MSG(res >= 0, "There is an error when doing socket select %s %d\n", strerror(errno), errno);

	return res;
}

int _starpu_tcpip_common_notif_recv_is_ready(const struct _starpu_mp_node *mp_node)
{
	fd_set set;
	int fd = mp_node->mp_connection.tcpip_mp_connection->notif_sock;
	int res;

	struct timeval tv =
	{
		.tv_sec = 0,
		.tv_usec = 0
	};

	FD_ZERO(&set);
	FD_SET(fd, &set);

	while((res = select(fd+1, &set, NULL, NULL, &tv)) == -1 && errno == EINTR);

	STARPU_ASSERT_MSG(res >= 0, "There is an error when doing socket select %s %d\n", strerror(errno), errno);

	return res;
}

int _starpu_tcpip_common_notif_send_is_ready(const struct _starpu_mp_node *mp_node)
{
	fd_set set;
	int fd = mp_node->mp_connection.tcpip_mp_connection->notif_sock;
	int res;

	struct timeval tv =
	{
		.tv_sec = 0,
		.tv_usec = 0
	};

	FD_ZERO(&set);
	FD_SET(fd, &set);

	while((res = select(fd+1, NULL, &set, NULL, &tv)) == -1 && errno == EINTR);

	STARPU_ASSERT_MSG(res >= 0, "There is an error when doing socket select %s %d\n", strerror(errno), errno);

	return res;
}

void _starpu_tcpip_common_wait(const struct _starpu_mp_node *mp_node)
{
	fd_set reads;
	fd_set writes;
	int fd_sync = mp_node->mp_connection.tcpip_mp_connection->sync_sock;
	int fd_notif = mp_node->mp_connection.tcpip_mp_connection->notif_sock;
	int fd_max = 0;
	int res;

	FD_ZERO(&reads);
	FD_ZERO(&writes);

	FD_SET(fd_sync, &reads);
	if(fd_sync > fd_max)
		fd_max = fd_sync;

	sigset_t sigmask;
	sigemptyset(&sigmask);

	STARPU_PTHREAD_MUTEX_LOCK(&mp_node->message_queue_mutex);
	if(!mp_message_list_empty(&mp_node->message_queue) || !_starpu_mp_event_list_empty(&mp_node->event_queue))
	{
		FD_SET(fd_notif, &writes);
		if(fd_notif > fd_max)
			fd_max = fd_notif;
	}
	STARPU_PTHREAD_MUTEX_UNLOCK(&mp_node->message_queue_mutex);

	res = pselect(fd_max+1, &reads, &writes, NULL, NULL, &sigmask);
	if(res < 0)
	STARPU_ASSERT_MSG(errno == EINTR, "There is an error when doing socket pselect %s %d\n", strerror(errno), errno);
}

void _starpu_tcpip_common_signal(const struct _starpu_mp_node *mp_node STARPU_ATTRIBUTE_UNUSED)
{
	int res;
	res = pthread_kill(master_thread, SIGUSR1);

	STARPU_ASSERT(res == 0);
}

static void __starpu_tcpip_common_send(const struct _starpu_mp_node *node, void *msg, int len, void * event, int notif);
static void __starpu_tcpip_common_recv(const struct _starpu_mp_node *node, void *msg, int len, void * event, int notif);
static void _starpu_tcpip_common_action_socket(what_t what, const char * whatstr, int is_sender, const struct _starpu_mp_node *node, struct _starpu_tcpip_socket *remote_sock, void *msg, int len, void * event, int notif);
static void _starpu_tcpip_common_send_to_socket(const struct _starpu_mp_node *node, struct _starpu_tcpip_socket *dst_sock, void *msg, int len, void * event, int notif);
static void _starpu_tcpip_common_recv_from_socket(const struct _starpu_mp_node *node, struct _starpu_tcpip_socket *src_sock, void *msg, int len, void * event, int notif);

/* SEND */
void _starpu_tcpip_common_mp_send(const struct _starpu_mp_node *node, void *msg, int len)
{
	__starpu_tcpip_common_send(node, msg, len, NULL, 0);
}

void _starpu_tcpip_common_nt_send(const struct _starpu_mp_node *node, void *msg, int len)
{
	__starpu_tcpip_common_send(node, msg, len, NULL, 1);
}

/* SEND to source node */
void _starpu_tcpip_common_send(const struct _starpu_mp_node *node, void *msg, int len, void * event)
{
	__starpu_tcpip_common_send(node, msg, len, event, 0);
}

static void __starpu_tcpip_common_send(const struct _starpu_mp_node *node, void *msg, int len, void * event, int notif)
{
	_starpu_tcpip_common_send_to_socket(node, node->mp_connection.tcpip_mp_connection, msg, len, event, notif);
}

/* SEND to any node */
void _starpu_tcpip_common_send_to_device(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, int devid, void *msg, int len, void * event)
{
	struct _starpu_tcpip_socket *dst_sock = &tcpip_sock[devid];
	_starpu_tcpip_common_send_to_socket(node, dst_sock, msg, len, event, 0);
}

static void _starpu_tcpip_common_send_to_socket(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, struct _starpu_tcpip_socket *dst_sock, void *msg, int len, void * event, int notif)
{
	_starpu_tcpip_common_action_socket((what_t)write, "send", 1, node, dst_sock, msg, len, event, notif);
}


/* RECV */
void _starpu_tcpip_common_mp_recv(const struct _starpu_mp_node *node, void *msg, int len)
{
	__starpu_tcpip_common_recv(node, msg, len, NULL, 0);
}

void _starpu_tcpip_common_nt_recv(const struct _starpu_mp_node *node, void *msg, int len)
{
	__starpu_tcpip_common_recv(node, msg, len, NULL, 1);
}

void _starpu_tcpip_common_recv(const struct _starpu_mp_node *node, void *msg, int len, void * event)
{
	__starpu_tcpip_common_recv(node, msg, len, event, 0);
}

/* RECV from source node */
static void __starpu_tcpip_common_recv(const struct _starpu_mp_node *node, void *msg, int len, void * event, int notif)
{
	_starpu_tcpip_common_recv_from_socket(node, node->mp_connection.tcpip_mp_connection, msg, len, event, notif);
}

/* RECV from any node */
void _starpu_tcpip_common_recv_from_device(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, int devid, void *msg, int len, void * event)
{
	struct _starpu_tcpip_socket *src_sock = &tcpip_sock[devid];
	_starpu_tcpip_common_recv_from_socket(node, src_sock, msg, len, event, 0);
}

static void _starpu_tcpip_common_recv_from_socket(const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, struct _starpu_tcpip_socket *src_sock, void *msg, int len, void * event, int notif)
{
	_starpu_tcpip_common_action_socket(read, "recv", 0, node, src_sock, msg, len, event, notif);
}

/*do refactor for SEND to and RECV from socket */
static void _starpu_tcpip_common_action_socket(what_t what, const char * whatstr, int is_sender, const struct _starpu_mp_node *node STARPU_ATTRIBUTE_UNUSED, struct _starpu_tcpip_socket *remote_sock, void *msg, int len, void * event, int notif)
{
	if (event)
	{
		_TCPIP_PRINT("async %s\n", whatstr);
		_TCPIP_PRINT("%s %d bytes to %d message %x\n", whatstr, len, remote_sock->async_sock, *((int *) (uintptr_t)msg));
		/* Asynchronous*/
		struct _starpu_async_channel * channel = event;
		struct _starpu_tcpip_ms_async_event *tcpip_ms_event = _starpu_tcpip_ms_async_event(&channel->event);
		tcpip_ms_event->is_sender = is_sender;

		/* call by sink, we need to initialize some parts, for host it's done in data_request.c */
		if (channel->node_ops == NULL)
			tcpip_ms_event->requests = NULL;

		/* Initialize the list */
		if (tcpip_ms_event->requests == NULL)
		{
			_STARPU_MALLOC(tcpip_ms_event->requests, sizeof(*tcpip_ms_event->requests));
			_starpu_tcpip_ms_request_multilist_head_init_event(tcpip_ms_event->requests);
		}

		struct _starpu_tcpip_ms_request *req;
		_STARPU_MALLOC(req, sizeof(*req));
		_starpu_tcpip_ms_request_multilist_init_thread(req);
		_starpu_tcpip_ms_request_multilist_init_event(req);
		_starpu_tcpip_ms_request_multilist_init_pending(req);

#ifdef STARPU_SANITIZE_ADDRESS
		/* Poke data immediately, to get a good backtrace where bogus
		 * pointers come from */
		if (is_sender)
		{
			char *c = malloc(len);
			memcpy(c, msg, len);
			free(c);
		}
		else
			memset(msg, 0, len);
#endif
		/*complete the fields*/
		req->remote_sock = remote_sock;
		req->len = len;
		req->buf = msg;
		req->flag_completed = 0;
		STARPU_HG_DISABLE_CHECKING(req->flag_completed);
		starpu_sem_init(&req->sem_wait_request, 0, 0);
		req->is_sender = is_sender;
		req->offset = 0;
		req->send_end = 0;

		_SELECT_PRINT("%s push back\n", whatstr);
		_starpu_spin_lock(&ListLock);
		_starpu_tcpip_ms_request_multilist_push_back_thread(&thread_list, req);
		_starpu_spin_unlock(&ListLock);

		char buf = 0;
		int res;
		while((res = write(thread_pipe[1], &buf, 1)) == -1 && errno == EINTR)
		;

		channel->starpu_mp_common_finished_receiver++;
		channel->starpu_mp_common_finished_sender++;

		_starpu_tcpip_ms_request_multilist_push_back_event(tcpip_ms_event->requests, req);
	}
	else
	{
		_TCPIP_PRINT("sync %s\n", whatstr);
		/* Synchronous send */
		if(!notif)
		{
			_TCPIP_PRINT("dst_sock is %d\n", remote_sock->sync_sock);
			int res, offset = 0;
			while(offset < len)
			{
				while((res = what(remote_sock->sync_sock, (char*)msg+offset, len-offset)) == -1 && errno == EINTR)
				;
				_TCPIP_PRINT("msg after write is %x, res is %d\n", *((int *) (uintptr_t)msg), res);
				STARPU_ASSERT_MSG(res != 0 && !(res == -1 && errno == ECONNRESET), "TCP/IP Master/Slave noticed that %s (peer %d) has exited unexpectedly", node->kind == STARPU_NODE_TCPIP_SOURCE ? "the master" : "some slave", node->peer_id);
				STARPU_ASSERT_MSG(res > 0, "TCP/IP Master/Slave cannot %s a msg synchronous with a size of %d Bytes!, the result of %s is %d, the error is %s ", whatstr, len, whatstr, res, strerror(errno));
				offset+=res;
			}
		}
		else
		{
			_TCPIP_PRINT("dst_sock is %d\n", remote_sock->notif_sock);
			int res, offset = 0;
			while(offset < len)
			{
				while((res = what(remote_sock->notif_sock, (char*)msg+offset, len-offset)) == -1 && errno == EINTR)
				;
				_TCPIP_PRINT("msg after write is %x, res is %d\n", *((int *) (uintptr_t)msg), res);
				STARPU_ASSERT_MSG(res != 0 && !(res == -1 && errno == ECONNRESET), "TCP/IP Master/Slave noticed that %s (peer %d) has exited unexpectedly", node->kind == STARPU_NODE_TCPIP_SOURCE ? "the master" : "some slave", node->peer_id);
				STARPU_ASSERT_MSG(res > 0, "TCP/IP Master/Slave cannot %s a msg notification with a size of %d Bytes!, the result of %s is %d, the error is %s ", whatstr, len, whatstr, res, strerror(errno));
				offset+=res;
			}
		}

		_TCPIP_PRINT("finish sync send\n");
	}

}

static void _starpu_tcpip_common_polling_node(struct _starpu_mp_node * node)
{
	/* poll the asynchronous messages.*/
	if (node != NULL)
	{
		STARPU_PTHREAD_MUTEX_LOCK(&node->connection_mutex);
		while(node->nt_recv_is_ready(node))
		{
			enum _starpu_mp_command answer;
			void *arg;
			int arg_size;
			//_TCPIP_PRINT("polling_node\n");
			answer = _starpu_nt_common_recv_command(node, &arg, &arg_size);
			if(!_starpu_src_common_store_message(node,arg,arg_size,answer))
			{
				_STARPU_ERROR("incorrect command '%s'", _starpu_mp_common_command_to_string(answer));
			}
		}
		STARPU_PTHREAD_MUTEX_UNLOCK(&node->connection_mutex);
	}
}

/*do refactor for test event and wait request completion */
static unsigned int _starpu_tcpip_common_action_completion(int wait, struct _starpu_async_channel * event)
{
	struct _starpu_tcpip_ms_async_event *tcpip_ms_event = _starpu_tcpip_ms_async_event(&event->event);

	if (tcpip_ms_event->requests != NULL)
	{
		struct _starpu_tcpip_ms_request * req;
		struct _starpu_tcpip_ms_request * req_next;

		//_TCPIP_PRINT("event requests is %p\n", req);
		for (req = _starpu_tcpip_ms_request_multilist_begin_event(tcpip_ms_event->requests);
		     req != _starpu_tcpip_ms_request_multilist_end_event(tcpip_ms_event->requests);
		     req = req_next)
		{
			req_next = _starpu_tcpip_ms_request_multilist_next_event(req);

			int flag = 0;
			if(!wait)
				flag = req->flag_completed;

			//_TCPIP_PRINT("the operation is finished? %d\n", flag);
			/*operation completed*/
			if (flag || wait)
			{
				starpu_sem_wait(&req->sem_wait_request);
				_starpu_tcpip_ms_request_multilist_erase_event(tcpip_ms_event->requests, req);
				STARPU_HG_ENABLE_CHECKING(req->flag_completed);
				free(req);

				if (tcpip_ms_event->is_sender)
					event->starpu_mp_common_finished_sender--;
				else
					event->starpu_mp_common_finished_receiver--;

				//_TCPIP_PRINT("common finished sender is %d\n", event->starpu_mp_common_finished_sender);
				//_TCPIP_PRINT("common finished receiver is %d\n", event->starpu_mp_common_finished_receiver);

			}

		}

		/* When the list is empty, we finished to wait each request */
		if (_starpu_tcpip_ms_request_multilist_empty_event(tcpip_ms_event->requests))
		{
			/* Destroy the list */
			free(tcpip_ms_event->requests);
			tcpip_ms_event->requests = NULL;
		}
	}

	//incoming ack from devices
	int i = 0;
	while((!wait && i++ == 0)||(wait && event->starpu_mp_common_finished_sender > 0) || (wait && event->starpu_mp_common_finished_receiver > 0))
	{
		_starpu_tcpip_common_polling_node(event->polling_node_sender);
		_starpu_tcpip_common_polling_node(event->polling_node_receiver);
	}

	if(!wait)
	return !event->starpu_mp_common_finished_sender && !event->starpu_mp_common_finished_receiver;
	else
	return 0;
}

/* - In device to device communications, the first ack received by host
 * is considered as the sender (but it cannot be, in fact, the sender)
 */
unsigned int _starpu_tcpip_common_test_event(struct _starpu_async_channel * event)
{
	return _starpu_tcpip_common_action_completion(0, event);
}

/* - In device to device communications, the first ack received by host
 * is considered as the sender (but it cannot be, in fact, the sender)
 */
void _starpu_tcpip_common_wait_request_completion(struct _starpu_async_channel * event)
{
	_starpu_tcpip_common_action_completion(1, event);
}

void _starpu_tcpip_common_barrier(void)
{
	char buf = 0;
	//_TCPIP_PRINT("index_sink (in common barrier) is %d\n", index_sink);
	int ret;
	/*master part*/
	if(index_sink == 0)
	{
		int i;
		for(i=1; i<nb_sink+1; i++)
		{
			//_TCPIP_PRINT("slave socket in sock list is %d\n", sock_list[i]);
			ret=read(tcpip_sock[i].sync_sock, &buf, 1);
			//printf("ret2 is %d\n", ret);
			STARPU_ASSERT_MSG(ret > 0, "Cannot read from slave!");
		}

		for(i=1; i<nb_sink+1; i++)
		{
			ret=write(tcpip_sock[i].sync_sock, &buf, 1);
			//printf("ret3 is %d\n", ret);
			STARPU_ASSERT_MSG(ret > 0, "Cannot write to slave!");
		}

	}
	/*slave part*/
	else
	{
		//_TCPIP_PRINT("master socket in sock list is %d\n", sock_list[0]);
		ret=write(tcpip_sock[0].sync_sock, &buf, 1);
		//printf("ret1 is %d\n", ret);
		STARPU_ASSERT_MSG(ret > 0, "Cannot write to master!");
		ret=read(tcpip_sock[0].sync_sock, &buf, 1);
		//printf("ret4 is %d\n", ret);
		STARPU_ASSERT_MSG(ret > 0, "Cannot read from master!");
	}
	_TCPIP_PRINT("finish common barrier\n");
}

/* Compute bandwidth and latency between source and sink nodes
 * Source node has to have the entire set of times at the end
 */
void _starpu_tcpip_common_measure_bandwidth_latency(double timing_dtod[STARPU_MAXTCPIPDEVS][STARPU_MAXTCPIPDEVS], double latency_dtod[STARPU_MAXTCPIPDEVS][STARPU_MAXTCPIPDEVS])
{
	int ret;
	unsigned iter;
	//_TCPIP_PRINT("index_sink is %d\n", index_sink);
	char * buf;
	_STARPU_MALLOC(buf, SIZE_BANDWIDTH);
	memset(buf, 0, SIZE_BANDWIDTH);

	_starpu_tcpip_common_mp_init();

	int sender, receiver;
	for(sender = 0; sender < nb_sink+1; sender++)
	{
		for(receiver = 0; receiver < nb_sink+1; receiver++)
		{
			//Node can't be a sender and a receiver
			if(sender == receiver)
				continue;

			if (!index_sink)
				_STARPU_DISP("measuring from %d to %d\n", sender, receiver);

			_starpu_tcpip_common_barrier();

			// _TCPIP_PRINT("sender id is %d\n", sender);
			// _TCPIP_PRINT("index_sink is %d\n", index_sink);
			if(index_sink == sender)
			{

				//_TCPIP_PRINT("sender id is %d\n", sender);
				double start, end;
				/* measure bandwidth sender to receiver */
				start = starpu_timing_now();
				for (iter = 0; iter < NITER; iter++)
				{
					ret = write(tcpip_sock[receiver].sync_sock, buf, SIZE_BANDWIDTH);
					STARPU_ASSERT_MSG(ret == SIZE_BANDWIDTH, "short write!");
					STARPU_ASSERT_MSG(ret > 0, "Bandwidth of TCP/IP Master/Slave cannot be measured !");
					ret = read(tcpip_sock[receiver].sync_sock, buf, 1);
					STARPU_ASSERT_MSG(ret > 0, "Bandwidth of TCP/IP Master/Slave cannot be measured !");
				}
				end = starpu_timing_now();
				timing_dtod[sender][receiver] = (end - start)/NITER/SIZE_BANDWIDTH;

				/* measure latency sender to receiver */
				start = starpu_timing_now();
				for (iter = 0; iter < NITER; iter++)
				{
					ret = write(tcpip_sock[receiver].sync_sock, buf, 1);
					STARPU_ASSERT_MSG(ret > 0, "Bandwidth of TCP/IP Master/Slave cannot be measured !");
					ret = read(tcpip_sock[receiver].sync_sock, buf, 1);
					STARPU_ASSERT_MSG(ret > 0, "Bandwidth of TCP/IP Master/Slave cannot be measured !");
				}
				end = starpu_timing_now();
				latency_dtod[sender][receiver] = (end - start)/NITER/2;
			}

			// _TCPIP_PRINT("receiver id is %d\n", receiver);
			// _TCPIP_PRINT("index_sink is %d\n", index_sink);
			if (index_sink == receiver)
			{

				//_TCPIP_PRINT("receiver id is %d\n", receiver);
				/* measure bandwidth sender to receiver*/
				for (iter = 0; iter < NITER; iter++)
				{
					size_t pending = SIZE_BANDWIDTH;
					while (pending)
					{
						ret = read(tcpip_sock[sender].sync_sock, buf, SIZE_BANDWIDTH);
						STARPU_ASSERT_MSG(ret > 0, "Bandwidth of TCP/IP Master/Slave cannot be measured !");
						pending -= ret;
					}
					ret = write(tcpip_sock[sender].sync_sock, buf, 1);
					STARPU_ASSERT_MSG(ret > 0, "Bandwidth of TCP/IP Master/Slave cannot be measured !");
				}

				/* measure latency sender to receiver */
				for (iter = 0; iter < NITER; iter++)
				{
					ret = read(tcpip_sock[sender].sync_sock, buf, 1);
					STARPU_ASSERT_MSG(ret > 0, "Bandwidth of TCP/IP Master/Slave cannot be measured !");
					ret = write(tcpip_sock[sender].sync_sock, buf, 1);
					STARPU_ASSERT_MSG(ret > 0, "Bandwidth of TCP/IP Master/Slave cannot be measured !");
				}
			}
		}

		/* When a sender finished its work, it has to send its results to the master */

		/* Master doesn't need to send to itself its data */
		if (sender == 0)
			goto print;

		/* if we are the sender, we send the data */
		if (sender == index_sink)
		{
			write(tcpip_sock[0].sync_sock, timing_dtod[sender], sizeof(timing_dtod[sender]));
			write(tcpip_sock[0].sync_sock, latency_dtod[sender], sizeof(latency_dtod[sender]));
		}

		/* the master node receives the data */
		if (index_sink == 0)
		{
			read(tcpip_sock[sender].sync_sock, timing_dtod[sender], sizeof(timing_dtod[sender]));
			read(tcpip_sock[sender].sync_sock, latency_dtod[sender], sizeof(latency_dtod[sender]));
		}

print:
		if (index_sink == 0)
		{
			for(receiver = 0; receiver < nb_sink+1; receiver++)
			{
				if(sender == receiver)
					continue;

				_STARPU_DISP("BANDWIDTH %d -> %d %.0fMB/s %.2fus\n", sender, receiver, 1/timing_dtod[sender][receiver], latency_dtod[sender][receiver]);
			}
		}
	}
	free(buf);
}
