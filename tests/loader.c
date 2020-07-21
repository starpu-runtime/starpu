/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

#include <common/config.h>

#include <sys/types.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <sys/resource.h>
#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <signal.h>
#include <string.h>

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)
#include <windows.h>
#else
#include <sys/time.h>
#endif

#ifdef STARPU_QUICK_CHECK
/* Quick checks are supposed to be real quick, typically less than 1s each, sometimes 10s */
#define  DEFAULT_TIMEOUT       60
#elif !defined(STARPU_LONG_CHECK)
/* Normal checks are supposed to be short enough, typically less than 10s each, sometimes 1-2m */
#define  DEFAULT_TIMEOUT       300
#else
/* Long checks can be very long */
#define  DEFAULT_TIMEOUT       1000
#endif
#define  AUTOTEST_SKIPPED_TEST 77

static pid_t child_pid = 0;
static int   timeout;

#if defined(_WIN32) && !defined(__MINGW32__) && !defined(__CYGWIN__)
static int mygettimeofday(struct timeval *tv, void *tz)
{
	if (tv)
	{
		FILETIME ft;
		unsigned long long res;
		GetSystemTimeAsFileTime(&ft);
		/* 100-nanosecond intervals since January 1, 1601 */
		res = ft.dwHighDateTime;
		res <<= 32;
		res |= ft.dwLowDateTime;
		res /= 10;
		/* Now we have microseconds */
		res -= (((1970-1601)*365) + 89) * 24ULL * 3600ULL * 1000000ULL;
		/* Now we are based on epoch */
		tv->tv_sec = res / 1000000ULL;
		tv->tv_usec = res % 1000000ULL;
	}
}
#else
#define mygettimeofday(tv,tz) gettimeofday(tv,tz)
#endif

#ifdef STARPU_GDB_PATH
static int try_launch_gdb(const char *exe, const char *core)
{
# define GDB_ALL_COMMAND "thread apply all bt full"
# define GDB_COMMAND "bt full"
	int err;
	pid_t pid;
	struct stat st;
	const char *top_builddir;
	char *gdb;

	err = stat(core, &st);
	if (err != 0)
	{
		fprintf(stderr, "while looking for core file of %s: %s: %m\n",
			exe, core);
		return -1;
	}

	if (!(st.st_mode & S_IFREG))
	{
		fprintf(stderr, "%s: not a regular file\n", core);
		return -1;
	}

	top_builddir = getenv("top_builddir");

	pid = fork();
	switch (pid)
	{
	case 0:					  /* kid */
		if (top_builddir != NULL)
		{
			/* Run gdb with Libtool.  */
			gdb = alloca(strlen(top_builddir)
				     + sizeof("/libtool") + 1);
			strcpy(gdb, top_builddir);
			strcat(gdb, "/libtool");
			err = execl(gdb, "gdb", "--mode=execute",
				    STARPU_GDB_PATH, "--batch",
				    "-ex", GDB_COMMAND,
				    "-ex", GDB_ALL_COMMAND,
				    exe, core, NULL);
		}
		else
		{
			/* Run gdb directly  */
			gdb = STARPU_GDB_PATH;
			err = execl(gdb, "gdb", "--batch",
				    "-ex", GDB_COMMAND,
				    "-ex", GDB_ALL_COMMAND,
				    exe, core, NULL);
		}
		if (err != 0)
		{
			fprintf(stderr, "while launching `%s': %m\n", gdb);
			exit(EXIT_FAILURE);
		}
		exit(EXIT_SUCCESS);
		break;

	case -1:
		fprintf(stderr, "fork: %m\n");
		return -1;

	default:				  /* parent */
		{
			pid_t who;
			int status;
			who = waitpid(pid, &status, 0);
			if (who != pid)
				fprintf(stderr, "while waiting for gdb "
					"process %d: %m\n", pid);
		}
	}
	return 0;
# undef GDB_COMMAND
# undef GDB_ALL_COMMAND
}
#endif	/* STARPU_GDB_PATH */

static void launch_gdb(const char *exe)
{
#ifdef STARPU_GDB_PATH
	char s[32];
	snprintf(s, sizeof(s), "core.%d", child_pid);
	if (try_launch_gdb(exe, s) < 0)
		try_launch_gdb(exe, "core");
#endif	/* STARPU_GDB_PATH */
}

static char *test_name;

static void test_cleaner(int sig)
{
	pid_t child_gid;
	int status;
	(void) sig;

	// send signal to all loader family members
	fprintf(stderr, "[error] test %s has been blocked for %d seconds. Mark it as failed\n", test_name, timeout);
	child_gid = getpgid(child_pid);
	kill(-child_gid, SIGQUIT);
	waitpid(child_pid, &status, 0);
	launch_gdb(test_name);
	raise(SIGQUIT);
	exit(EXIT_FAILURE);
}

static void forwardsig(int sig)
{
	pid_t child_gid;
	child_gid = getpgid(child_pid);
	kill(-child_gid, sig);
}

static int _decode(char **src, char *motif, const char *value)
{
	char *found;

	found = strstr(*src, motif);
	if (found == NULL) return 0;

	char *new_src = calloc(1, strlen(*src)-strlen(motif)+strlen(value)+1);

	strncpy(new_src, *src, found - *src);
	strcat(new_src, value);
	strcat(new_src, found+strlen(motif));

	*src = new_src;
	return 1;
}

static void decode(char **src, char *motif, const char *value)
{
	if (*src)
	{
		if (strstr(*src, motif) && value == NULL)
		{
			fprintf(stderr, "error: $%s undefined\n", motif);
			exit(EXIT_FAILURE);
		}
		int d = _decode(src, motif, value);
		while (d)
			d = _decode(src, motif, value);
	}
}

int main(int argc, char *argv[])
{
	int   child_exit_status;
	char *test_args;
	char *launcher;
	char *launcher_args;
	char *libtool;
	const char *top_builddir = getenv ("top_builddir");
	struct sigaction sa;
	int   ret;
	struct timeval start;
	struct timeval end;
	double timing;
	int x=1;

	(void) argc;
	test_args = NULL;
	timeout = 0;

	launcher=getenv("STARPU_CHECK_LAUNCHER");
	launcher_args=getenv("STARPU_CHECK_LAUNCHER_ARGS");

	if (argv[x] && strcmp(argv[x], "-t") == 0)
	{
		timeout = strtol(argv[x+1], NULL, 10);
		x += 2;
	}
	else if (getenv("STARPU_TIMEOUT_ENV"))
	{
		/* get user-defined iter_max value */
		timeout = strtol(getenv("STARPU_TIMEOUT_ENV"), NULL, 10);
	}
	if (timeout <= 0)
	{
		timeout = DEFAULT_TIMEOUT;
		if ((launcher && strstr(launcher, "valgrind")) ||
		    (launcher && strstr(launcher, "helgrind")) ||
		    getenv("TSAN_OPTIONS") != NULL)
			timeout *= 20;
		if (getenv("ASAN_OPTIONS") != NULL ||
		    getenv("USAN_OPTIONS") != NULL ||
		    getenv("LSAN_OPTIONS") != NULL)
			timeout *= 5;
	}

#ifdef STARPU_SIMGRID
	timeout *= 10;
#endif

#ifdef STARPU_USE_MPI_MASTER_SLAVE
	/* compare values between the 2 values of timeout */
	if (getenv("MPIEXEC_TIMEOUT"))
	{
		int mpiexec_timeout = strtol(getenv("MPIEXEC_TIMEOUT"), NULL, 10);
		if (mpiexec_timeout != timeout)
			fprintf(stderr, "[warning] MPIEXEC_TIMEOUT and STARPU_TIMEOUT_ENV values are different (%d and %d). The behavior may be different than expected !\n", mpiexec_timeout, timeout);
	}
#endif

	if (argv[x] && strcmp(argv[x], "-p") == 0)
	{
		test_name = malloc(strlen(argv[x+1]) + 1 + strlen(argv[x+2]) + 1);
		sprintf(test_name, "%s/%s", argv[x+1], argv[x+2]);
		x += 3;
	}
	else
	{
		test_name = argv[x];
		x += 1;
	}

	if (!test_name)
	{
		fprintf(stderr, "[error] Need name of program to start\n");
		exit(EXIT_FAILURE);
	}

	size_t len = strlen(test_name);
	if (len >= 3 &&
	    test_name[len-3] == '.' &&
	    test_name[len-2] == 's' &&
	    test_name[len-1] == 'h')
	{
                /* This is a shell script, don't run ourself on bash, but make
                 * the script call us for each program invocation */

		setenv("STARPU_LAUNCH", argv[0], 1);

		execvp(test_name, argv+x-1);

		fprintf(stderr, "[error] '%s' failed to exec. test marked as failed\n", test_name);
		exit(EXIT_FAILURE);
	}

	if (strstr(test_name, "spmv/dw_block_spmv"))
	{
		test_args = (char *) calloc(512, sizeof(char));
		snprintf(test_args, 512, "%s/examples/spmv/matrix_market/examples/fidapm05.mtx", STARPU_SRC_DIR);
	}
	else if (strstr(test_name, "starpu_perfmodel_display"))
	{
		if (x >= argc)
			test_args = strdup("-l");
	}
	else if (strstr(test_name, "starpu_perfmodel_plot"))
	{
		if (x >= argc)
			test_args = strdup("-l");
	}

	/* get launcher program */
	if (launcher_args)
		launcher_args=strdup(launcher_args);

	if (top_builddir == NULL)
	{
		fprintf(stderr,
			"warning: $top_builddir undefined, "
			"so $STARPU_CHECK_LAUNCHER ignored\n");
		launcher = NULL;
		launcher_args = NULL;
		libtool = NULL;
	}
	else
	{
		libtool = malloc(strlen(top_builddir) + 1 + strlen("libtool") + 1);
		strcpy(libtool, top_builddir);
		strcat(libtool, "/libtool");
	}

	if (launcher)
	{
		const char *top_srcdir = getenv("top_srcdir");
		decode(&launcher, "@top_srcdir@", top_srcdir);
		decode(&launcher_args, "@top_srcdir@", top_srcdir);
	}

	setenv("STARPU_OPENCL_PROGRAM_DIR", STARPU_SRC_DIR, 1);

	/* set SIGALARM handler */
	sa.sa_flags = 0;
	sigemptyset(&sa.sa_mask);
	sa.sa_handler = test_cleaner;
	if (-1 == sigaction(SIGALRM, &sa, NULL))
		perror("sigaction");

	signal(SIGINT, forwardsig);
	signal(SIGHUP, forwardsig);
	signal(SIGPIPE, forwardsig);
	signal(SIGTERM, forwardsig);

	child_pid = fork();
	if (child_pid == 0)
	{
		char *launcher_argv[100];
		int i=0;

		setpgid(0, 0);

		/* "Launchers" such as Valgrind need to be inserted
		 * after the Libtool-generated wrapper scripts, hence
		 * this special-case.  */
		if (launcher && top_builddir != NULL)
		{
			launcher_argv[i++] = libtool;
			launcher_argv[i++] = "--mode=execute";
			launcher_argv[i++] = launcher;
			if (launcher_args)
			{
				launcher_argv[i++] = strtok(launcher_args, " ");
				while (launcher_argv[i-1])
				{
					launcher_argv[i++] = strtok(NULL, " ");
				}
			}
		}

		launcher_argv[i++] = test_name;
		if (test_args)
			launcher_argv[i++] = test_args;
		else while (argv[x])
		{
			launcher_argv[i++] = argv[x++];
		}
#ifdef STARPU_SIMGRID
		launcher_argv[i++] = "--cfg=contexts/factory:thread";
#endif
		launcher_argv[i++] = NULL;
		execvp(*launcher_argv, launcher_argv);

		fprintf(stderr, "[error] '%s' failed to exec. test marked as failed\n", test_name);
		exit(EXIT_FAILURE);
	}
	if (child_pid == -1)
	{
		fprintf(stderr, "[error] fork. test marked as failed\n");
		exit(EXIT_FAILURE);
	}
	free(test_args);
	free(libtool);

	ret = EXIT_SUCCESS;
	gettimeofday(&start, NULL);
	alarm(timeout);
	if (child_pid == waitpid(child_pid, &child_exit_status, 0))
	{
		if (WIFEXITED(child_exit_status))
		{
			int status = WEXITSTATUS(child_exit_status);
			if (status == EXIT_SUCCESS)
			{
				alarm(0);
			}
			else
			{
				if (status != AUTOTEST_SKIPPED_TEST)
					fprintf(stdout, "`%s' exited with return code %d\n",
						test_name, status);
				ret = status;
			}
		}
		else if (WIFSIGNALED(child_exit_status))
		{
			fprintf(stderr, "[error] `%s' killed with signal %d; test marked as failed\n",
				test_name, WTERMSIG(child_exit_status));
			launch_gdb(test_name);
			ret = EXIT_FAILURE;
		}
		else
		{
			fprintf(stderr, "[error] `%s' did not terminate normally; test marked as failed\n",
				test_name);
			ret = EXIT_FAILURE;
		}
	}

	gettimeofday(&end, NULL);
	timing = (double)((end.tv_sec - start.tv_sec)*1000000 + (end.tv_usec - start.tv_usec));
	fprintf(stderr, "#Execution_time_in_seconds %f %s\n", timing/1000000, test_name);

	return ret;
}
