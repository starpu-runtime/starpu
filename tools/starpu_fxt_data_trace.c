/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2011-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2013       Joris Pablo
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

#include <stdio.h>
#include <starpu.h>
#include <string.h>
#include <sys/stat.h>
#include <common/config.h>

#define PROGNAME "starpu_fxt_data_trace"
#define MAX_LINE_SIZE 100

static void usage()
{
	fprintf(stderr, "Get statistics about tasks lengths and data size\n\n");
	fprintf(stderr, "Usage: %s [ options ] <filename> [<codelet1> <codelet2> .... <codeletn>]\n", PROGNAME);
        fprintf(stderr, "\n");
        fprintf(stderr, "Options:\n");
	fprintf(stderr, "   -h, --help          display this help and exit\n");
	fprintf(stderr, "   -v, --version       output version information and exit\n\n");
	fprintf(stderr, "    filename           specify the FxT trace input file.\n");
	fprintf(stderr, "    codeletX           specify the codelet name to profile (by default, all codelets are profiled)\n");
        fprintf(stderr, "Report bugs to <%s>.", PACKAGE_BUGREPORT);
        fprintf(stderr, "\n");
}

static int parse_args(int argc, char **argv)
{
	int i;

	if(argc < 2)
	{
		fprintf(stderr, "Incorrect usage, aborting\n");
                usage();
		return 77;
	}

	for (i = 1; i < argc; i++)
	{
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0)
		{
			usage();
			return EXIT_SUCCESS;
		}

		if (strcmp(argv[i], "-v") == 0 || strcmp(argv[i], "--version") == 0)
		{
		        fputs(PROGNAME " (" PACKAGE_NAME ") " PACKAGE_VERSION "\n", stderr);
			return EXIT_SUCCESS;
		}
	}
	return 0;
}

static void write_gp(int argc, char **argv)
{
	FILE *codelet_list = fopen("codelet_list", "r");
	if(!codelet_list)
	{
		perror("Error while opening codelet_list:");
		exit(-1);
	}
	char codelet_name[MAX_LINE_SIZE];
	const char *file_name = "data_trace.gp";
	FILE *plt = fopen(file_name, "w+");
	if(!plt)
	{
		perror("Error while creating data_trace.gp:");
		exit(-1);
	}

	fprintf(plt, "#!/usr/bin/gnuplot -persist\n\n");
	fprintf(plt, "set term postscript eps enhanced color\n");
	fprintf(plt, "set output \"data_trace.eps\"\n");
	fprintf(plt, "set title \"Data trace\"\n");
	fprintf(plt, "set logscale x\n");
	fprintf(plt, "set logscale y\n");
	fprintf(plt, "set xlabel \"data size (B)\"\n");
	fprintf(plt, "set ylabel \"tasks size (ms)\"\n");
	fprintf(plt, "plot ");
	int c_iter;
	char *v_iter;
	int begin = 1;
	while(fgets(codelet_name, MAX_LINE_SIZE, codelet_list) != NULL)
	{
		if(argc == 0)
		{
			if(begin)
				begin = 0;
			else
			fprintf(plt, ", ");
		}
		int size = strlen(codelet_name);
		if(size > 0)
			codelet_name[size-1] = '\0';
		if(argc != 0)
		{
			for(c_iter = 0, v_iter = argv[c_iter];
			    c_iter < argc;
			    c_iter++, v_iter = argv[c_iter])
			{
				if(!strcmp(v_iter, codelet_name))
				{
					if(begin)
						begin = 0;
					else
						fprintf(plt, ", ");
					fprintf(plt, "\"%s\" using 2:1 with dots lw 1 title \"%s\"", codelet_name, codelet_name);
				}
			}
		}
		else
		{
			fprintf(plt, "\"%s\" using 2:1 with dots lw 1 title \"%s\"", codelet_name, codelet_name);
		}
	}
	fprintf(plt, "\n");

	if(fclose(codelet_list))
	{
		perror("close failed :");
		exit(-1);
	}

	if(fclose(plt))
	{
		perror("close failed :");
		exit(-1);
	}

	struct stat sb;
	int ret = stat(file_name, &sb);
	if (ret)
	{
		perror("stat");
		STARPU_ABORT();
	}

	/* Make the gnuplot scrit executable for the owner */
	ret = chmod(file_name, sb.st_mode|S_IXUSR);
	if (ret)
	{
		perror("chmod");
		STARPU_ABORT();
	}
	fprintf(stdout, "Gnuplot file <data_trace.gp> has been successfully created.\n");
}

int main(int argc, char **argv)
{
	int ret = parse_args(argc, argv);
	if (ret) return ret;
	starpu_fxt_write_data_trace(argv[1]);
	write_gp(argc - 2, argv + 2);
	starpu_perfmodel_free_sampling();
	return 0;
}
