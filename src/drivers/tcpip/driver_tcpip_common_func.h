/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2021-2022-  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#define _TCPIP_DEBUG 0
#if _TCPIP_DEBUG
#  define _TCPIP_PRINT(...) printf(__VA_ARGS__)
#else
#  define _TCPIP_PRINT(...)
#endif

#ifdef __linux__
#include <linux/errno.h>
#ifndef ENOTSUPP
#define ENOTSUPP 524
#endif
#endif

enum errcase {SOCK_INIT, SOCK_GETADDRINFO, SOCK_GETADDRINFO_LOCAL};

#define SOCKET(domain, type, protocol, errcase) ({\
			int sock = 0;		  \
			sock = socket(domain, type, protocol);	\
			if(sock < 0)				\
			{					\
				if(errcase == SOCK_GETADDRINFO)	\
				{					\
					if (errno != EAFNOSUPPORT) /* do not raise exception if ipv6 is not available */ \
						perror("fail to create socket"); \
					return 1;			\
				}					\
				else if(errcase == SOCK_GETADDRINFO_LOCAL) \
				{					\
					if (errno != EAFNOSUPPORT) /* do not raise exception if ipv6 is not available */ \
						perror("fail to create socket"); \
					return -1;			\
				}					\
				else					\
				{					\
					perror("fail to create socket"); \
					return -1;			\
				}					\
			}						\
			sock;						\
		})

#define BIND(sockfd, addr, addrlen) ({ \
			if(bind(sockfd, addr, addrlen) != 0)	\
			{					\
				perror("socket fails to bind"); \
				return -1;			\
			}					\
		})

#define LISTEN(sockfd, backlog)({ \
			if(listen(sockfd, backlog) != 0)	\
			{					\
				perror("socket fails to listen");	\
				return -1;				\
			}						\
		})

#define ADDR_INIT(source_addr, source_port) ({ \
			struct sockaddr_in sockaddr_init; \
			memset(&sockaddr_init, 0, sizeof(sockaddr_init)); \
			sockaddr_init.sin_family = AF_INET;		\
			sockaddr_init.sin_addr.s_addr = source_addr;	\
			sockaddr_init.sin_port = source_port;		\
			sockaddr_init;					\
		})

#define LOCAL_ADDR_INIT(source_addr_init) ({ \
			struct sockaddr_un name;	\
			memset(&name, 0, sizeof(name)); \
			name.sun_family = AF_UNIX;			\
			snprintf(name.sun_path, sizeof(name.sun_path) - 1, "/tmp/starpu-%d.socket", ntohs(source_addr_init.sin_port)); \
			name;						\
		})

#define GETSOCKNAME(sockfd, addr, addrlen) ({ \
			if(getsockname(sockfd, addr, addrlen) != 0)	\
			{						\
				perror("getsockname fail");		\
				return -1;				\
			}						\
		})

#define GETPEERNAME(sockfd, addr, addrlen) ({ \
			if(getpeername(sockfd, addr, addrlen) != 0)	\
			{						\
				perror("getpeername fail");		\
				return -1;				\
			}						\
		})

#define ACCEPT(sockfd, addr, addrlen) ({ \
			int sock;		      \
			sock = accept(sockfd, addr, addrlen);	\
			if(sock < 0)				\
			{					\
				perror("fail to receive the request of slave"); \
				return -1;				\
			}						\
			sock;						\
		})

#define CONNECT(sockfd, addr, addrlen, cur) ({ \
			if (connect(sockfd, addr, addrlen) < 0) \
			{					\
				perror("fail to connect socket");	\
				close(sockfd);				\
				if(cur)					\
					return 1;			\
				else					\
					return -1;			\
			}						\
		})

#define WRITE(fd, buf, count) ({      \
			if(write(fd, buf, count) < 0)	\
			{				\
				perror("fail to send"); \
				return -1;		\
			}				\
		})

#define READ(fd, buf, count) ({ \
			if(read(fd, buf, count) < 0)	\
			{				\
				perror("fail to receive");	\
				return -1;			\
			}					\
		})

#define SETSOCKOPT_ZEROCOPY(sockfd, optname) ({ \
			int zc;			\
			int one = 1;					\
			int ret = setsockopt(sockfd, SOL_SOCKET, optname, &one, sizeof(one)); \
			if (ret!=0)					\
			{						\
				if (errno != EOPNOTSUPP && errno != ENOPROTOOPT && errno != ENOTSUPP) \
					perror("setsockopt zerocopy");	\
				zc = 0;					\
			}						\
			else						\
				zc = 1;					\
			zc;						\
		})


/* This function contains all steps to initialize a socket before connect and accept steps.
 * When we call this function, we need to indicate that it is for master-slave (master = 1)
 * or slave-slave (master = 0). We also need to provide the information sin_addr "source_addr"
 * and sin_port "source_port" that we want to set to initialize the binding address and
 * the argument "backlog" for listen. It can generate a TCP/IP socket "ss" or a local socket "ls",
 * and the bound address "source_addr_init" with its size "source_addr_init_size".
 * For local socket, it also generates the bound address "local_name" linking a local path.
 */
static inline int master_init(int master, int *ss, int *ls, struct sockaddr_in *source_addr_init, socklen_t *source_addr_init_size, struct sockaddr_un *local_name, unsigned long source_addr, unsigned short source_port, int backlog)
{
	/*TCPIP*/
	*ss = SOCKET(AF_INET, SOCK_STREAM, 0, SOCK_INIT);

	struct sockaddr_in addr_init = ADDR_INIT(source_addr, source_port);
	socklen_t addr_init_size = sizeof(addr_init);

	if(master)
	{
		int one = 1;
		setsockopt(*ss, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
	}


	BIND(*ss, (struct sockaddr*) &addr_init, addr_init_size);

	if(!master)
	{
		GETSOCKNAME(*ss, (struct sockaddr*) &addr_init, &addr_init_size);
	}

	LISTEN(*ss, backlog);

	*source_addr_init = addr_init;
	*source_addr_init_size = addr_init_size;

	/*local socket*/
	if (starpu_get_env_number_default("STARPU_TCPIP_USE_LOCAL_SOCKET", 1) != 0)
	{
		*ls = SOCKET(AF_UNIX, SOCK_STREAM, 0, 0);

		*local_name = LOCAL_ADDR_INIT(addr_init);

		_TCPIP_PRINT("local socket name is %s\n", local_name->sun_path);
		unlink(local_name->sun_path);

		BIND(*ls, (const struct sockaddr *) &(*local_name), sizeof(*local_name));

		LISTEN(*ls, backlog);
	}

	return 0;
}

/* Accept step. We provide the TCP/IP socket "source_sock" or local socket "local_sock"
 * which is ready to accept the connection request from the other side. It will generate
 * the socket of the other side "sink_sock". It will also show whether the zerocopy setting
 * is successful (zerocopy = 1) or not (zerocopy = 0). This setting is only for async communication.
 */
static inline int master_accept(int *sink_sock, int source_sock, int local_sock, int *zerocopy, int * local_sock_flag)
{
	struct sockaddr_in sink_addr;
	socklen_t sink_addr_size = sizeof(sink_addr);

	*sink_sock = ACCEPT(source_sock, (struct sockaddr*)&sink_addr, &sink_addr_size);

	if (zerocopy != NULL)
	{
	#ifdef SO_ZEROCOPY
		*zerocopy = SETSOCKOPT_ZEROCOPY(*sink_sock, SO_ZEROCOPY);
	#else
		*zerocopy = 0;
	#endif
	}

	if (local_sock_flag != NULL)
		*local_sock_flag = 0;

	/*local socket*/
	if (starpu_get_env_number_default("STARPU_TCPIP_USE_LOCAL_SOCKET", 1) != 0)
	{
		struct sockaddr_in boundAddr;
		socklen_t boundAddr_size = sizeof(boundAddr);

		GETSOCKNAME(*sink_sock, (struct sockaddr*) &boundAddr, &boundAddr_size);

		/*master and slave sides use the same ip address*/
		if(boundAddr.sin_addr.s_addr == sink_addr.sin_addr.s_addr)
		{
		    close(*sink_sock);
		    *sink_sock = ACCEPT(local_sock, NULL, NULL);

		    if (local_sock_flag != NULL)
			*local_sock_flag = 1;
		}

		if (zerocopy != NULL)
		{
		#ifdef SO_ZEROCOPY
			*zerocopy = SETSOCKOPT_ZEROCOPY(*sink_sock, SO_ZEROCOPY);
		#else
			*zerocopy = 0;
		#endif
		}
	}

	return 0;
}

/* Connect step. We provide the connection address for TCP/IP socket, either it is addrinfo "cur" got from
 * function getaddrinfo in master-salve mode, or it is "source_addr" in slave-slave mode. It will generate
 * the socket of the other side "source_sock", In the case that slave connects to master, we need to get
 * the address "source_addr" to which "source_sock" is bound. It will also show whether the zerocopy setting
 * is successful (zerocopy = 1) or not (zerocopy = 0). This setting is only for async communication.
 */
static inline int slave_connect(int *source_sock, struct addrinfo *cur, struct sockaddr_in *bound_addr, struct sockaddr_in *source_addr, int *zerocopy, int * local_sock_flag)
{
	if(cur != NULL)
	{
		*source_sock = SOCKET(cur->ai_family, cur->ai_socktype, cur->ai_protocol, SOCK_GETADDRINFO);
		CONNECT(*source_sock, cur->ai_addr, cur->ai_addrlen, 1);
	}
	else
	{
		*source_sock = SOCKET(AF_INET, SOCK_STREAM, 0, SOCK_INIT);
		CONNECT(*source_sock, (struct sockaddr*)&(*source_addr), sizeof(*source_addr), 0);
	}

	if (zerocopy != NULL)
	{
	#ifdef SO_ZEROCOPY
		*zerocopy = SETSOCKOPT_ZEROCOPY(*source_sock, SO_ZEROCOPY);
	#else
		*zerocopy = 0;
	#endif
	}

	if (local_sock_flag != NULL)
		*local_sock_flag = 0;

	struct sockaddr_in boundAddr, peerAddr;
	socklen_t boundAddr_size = sizeof(boundAddr);
	socklen_t peerAddr_size = sizeof(peerAddr);

	GETSOCKNAME(*source_sock, (struct sockaddr*) &boundAddr, &boundAddr_size);
	GETPEERNAME(*source_sock, (struct sockaddr*) &peerAddr, &peerAddr_size);

	if(bound_addr != NULL)
		*bound_addr = boundAddr;

	/*local socket*/
	if (starpu_get_env_number_default("STARPU_TCPIP_USE_LOCAL_SOCKET", 1) != 0)
	{
		/*master and slave sides use the same ip address*/
		if(boundAddr.sin_addr.s_addr == peerAddr.sin_addr.s_addr)
		{
			close(*source_sock);
			if(cur != NULL)
				*source_sock = SOCKET(AF_UNIX, SOCK_STREAM, 0, SOCK_GETADDRINFO_LOCAL);
			else
				*source_sock = SOCKET(AF_UNIX, SOCK_STREAM, 0, SOCK_INIT);

			struct sockaddr_un local_name = LOCAL_ADDR_INIT(peerAddr);

			_TCPIP_PRINT("local socket name %s is got for sync connect\n", local_name.sun_path);

			CONNECT(*source_sock, (const struct sockaddr *) &local_name, sizeof(local_name), 0);

			if (local_sock_flag != NULL)
				*local_sock_flag = 1;
		}

		if (zerocopy != NULL)
		{
		#ifdef SO_ZEROCOPY
			*zerocopy = SETSOCKOPT_ZEROCOPY(*source_sock, SO_ZEROCOPY);
		#else
			*zerocopy = 0;
		#endif
		}
	}

	return 0;
}
