/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2009-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
#include <starpu.h>
#include <stdio.h>
#if defined(_WIN32) && !defined(__CYGWIN__)
#include <windows.h>
#endif

#define PROGNAME "starpu_calibrate_bus"

static void usage(void)
{
		(void) fprintf(stdout,
"Force a bus calibration.\n\
\n\
Usage: %s [OPTION]\n\
\n\
Options:\n\
	-h, --help       display this help and exit\n\
	-v, --version    output version information and exit\n\
\n\
Report bugs to <%s>.\n", PROGNAME, PACKAGE_BUGREPORT);
}

static void parse_args(int argc, char **argv)
{
	if (argc == 1)
		return;

	if (argc > 2)
	{
		usage();
		exit(EXIT_FAILURE);
	}

	if (strcmp(argv[1], "-h") == 0 ||
	    strcmp(argv[1], "--help") == 0)
	{
		usage();
		exit(EXIT_SUCCESS);
	}
	else if (strcmp(argv[1], "-v") == 0 ||
		 strcmp(argv[1], "--version") == 0)
	{
	        fputs(PROGNAME " (" PACKAGE_NAME ") " PACKAGE_VERSION "\n", stderr);
		exit(EXIT_SUCCESS);
	}
	else
	{
		(void) fprintf(stderr, "Unknown arg %s\n", argv[1]);
		exit(EXIT_FAILURE);
	}

}

int main(int argc, char **argv)
{
	int ret;
	struct starpu_conf conf;

	parse_args(argc, argv);

	starpu_conf_init(&conf);
	conf.bus_calibrate = 1;

	ret = starpu_init(&conf);
	if (ret == -ENODEV) return 77;
	if (ret != 0) return ret;

	starpu_shutdown();

	return 0;
}
