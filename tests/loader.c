/*
 * PM2: Parallel Multithreaded Machine
 * Copyright (C) 2001 the PM2 team (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation; either version 2 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
 * General Public License for more details.
 */


#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>

#define  DEFAULT_TIMEOUT       250
#define  AUTOTEST_SKIPPED_TEST 77

static pid_t child_pid = 0;

static void test_cleaner(int sig)
{
	pid_t child_gid;

	// send signal to all loader family members
	child_gid = getpgid(child_pid);
	kill(-child_gid, SIGKILL);
	exit(EXIT_FAILURE);
}

int main(int argc, char *argv[])
{
	int   timeout;
	int   child_exit_status;
	char *test_name;
	int   status;
	struct sigaction sa;

	timeout = 0;
	test_name = argv[1];

	if (!test_name)
	{
		fprintf(stderr, "Error. Need name of program to start\n");
		exit(-1);
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
			exit(EXIT_FAILURE);
		}
		execl(test_name, test_name, NULL);
		exit(EXIT_FAILURE);
	}
	if (child_pid == -1)
		exit(EXIT_FAILURE);

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
					fprintf(stdout, "Exited with return code %d\n", status);
				return status;
			}
		}
		else
		{
			return EXIT_FAILURE;
		}
	}

	return EXIT_SUCCESS;
}
