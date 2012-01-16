/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010  Université de Bordeaux 1
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

#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>

#include <common/config.h>

#define  DEFAULT_TIMEOUT       600
#define  AUTOTEST_SKIPPED_TEST 77

static pid_t child_pid = 0;
static int   timeout;

static void test_cleaner(int sig)
{
	pid_t child_gid;

	// send signal to all loader family members
	fprintf(stderr, "[error] test is blocked since %d seconds. Mark it as failed\n", timeout);
	child_gid = getpgid(child_pid);
	kill(-child_gid, SIGKILL);
	exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
	int   child_exit_status;
	char *test_name;
	char *test_args;
	int   status;
	struct sigaction sa;

	test_args = NULL;
	timeout = 0;
	test_name = argv[1];

	if (!test_name)
	{
		fprintf(stderr, "[error] Need name of program to start\n");
		exit(EXIT_FAILURE);
	}

	if (strstr(test_name, "spmv/dw_block_spmv"))
	{
		test_args = (char *) malloc(150*sizeof(char));
		sprintf(test_args, "%s/examples/spmv/matrix_market/examples/fidapm05.mtx", STARPU_SRC_DIR);
	}

	/* get user-defined iter_max value */
	if (getenv("STARPU_TIMEOUT_ENV"))
		timeout = strtol(getenv("STARPU_TIMEOUT_ENV"), NULL, 10);
	if (timeout <= 0)
		timeout = DEFAULT_TIMEOUT;

	/* set SIGALARM handler */
	sa.sa_flags = 0;
	sigemptyset(&sa.sa_mask);
	sa.sa_handler = test_cleaner;
	if (-1 == sigaction(SIGALRM, &sa, NULL))
		perror("sigaction");

	child_pid = fork();
	if (child_pid == 0)
	{
		// get a new pgid
		if (setpgid(0, 0) == -1)
		{
			perror("setpgid");
			fprintf(stderr, "[error] setpgid. Mark test as failed\n");
			exit(EXIT_FAILURE);
		}
		execl(test_name, test_name, test_args, NULL);
		fprintf(stderr, "[error] execl. Mark test as failed\n");
		exit(EXIT_FAILURE);
	}
	if (child_pid == -1)
	{
		fprintf(stderr, "[error] fork. Mark test as failed\n");
		exit(EXIT_FAILURE);
	}

	alarm(timeout);
	if (child_pid == waitpid(child_pid, &child_exit_status, 0))
	{
		if (WIFEXITED(child_exit_status))
		{
			status = WEXITSTATUS(child_exit_status);
			if (status == EXIT_SUCCESS)
			{
				alarm(0);
			}
			else
			{
				if (status != AUTOTEST_SKIPPED_TEST)
					fprintf(stdout, "`%s' exited with return code %d\n",
						test_name, status);
				return status;
			}
		}
		else if (WIFSIGNALED(child_exit_status))
		{
			fprintf(stderr, "[error] `%s' killed with signal %d; test marked as failed\n",
				test_name, WTERMSIG(child_exit_status));
			return EXIT_FAILURE;
		}
		else
		{
			fprintf(stderr, "[error] `%s' did not terminate normally; test marked as failed\n",
				test_name);
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}
