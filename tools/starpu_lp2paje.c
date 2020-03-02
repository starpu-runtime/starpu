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

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <common/config.h>

#define PROGNAME "starpu_lp2paje"

struct task
{
	double start;
	double stop;
	int num;
	int worker;
};

int main(int argc, char *argv[])
{
	int nw, nt;
	double tmax;
	int i, w, ww, t, tt;
	int foo;
	double bar;

	if (argc != 1)
	{
		if (strcmp(argv[1], "-v") == 0 || strcmp(argv[1], "--version") == 0)
		{
			fprintf(stderr, "%s (%s) %s\n", PROGNAME, PACKAGE_NAME, PACKAGE_VERSION);
			exit(EXIT_SUCCESS);
		}
		fprintf(stderr, "Convert schedule optimized by lp into the Paje format\n\n");
		fprintf(stderr, "Usage: lp_solve file.lp | %s > paje.trace\n", PROGNAME);
		fprintf(stderr, "Report bugs to <%s>.", PACKAGE_BUGREPORT);
		fprintf(stderr, "\n");
		exit(EXIT_SUCCESS);
	}
	assert(scanf("Suboptimal solution\n") == 0);
	assert(scanf("\nValue of objective function: %lf\n", &tmax) == 1);

	assert(scanf("Actual values of the variables:\n") == 0);
	assert(scanf("tmax %lf\n", &tmax) == 1);
	assert(scanf("nt %d\n", &nt) == 1);
	assert(nt >= 0);
	assert(scanf("nw %d\n", &nw) == 1);
	assert(nw >= 0);
	printf(
"%%EventDef PajeDefineContainerType 1\n"
"%%  Alias         string\n"
"%%  ContainerType string\n"
"%%  Name          string\n"
"%%EndEventDef\n"
"%%EventDef PajeCreateContainer     2\n"
"%%  Time          date\n"
"%%  Alias         string\n"
"%%  Type          string\n"
"%%  Container     string\n"
"%%  Name          string\n"
"%%EndEventDef\n"
"%%EventDef PajeDefineStateType     3\n"
"%%  Alias         string\n"
"%%  ContainerType string\n"
"%%  Name          string\n"
"%%EndEventDef\n"
"%%EventDef PajeDestroyContainer    4\n"
"%%  Time          date\n"
"%%  Name          string\n"
"%%  Type          string\n"
"%%EndEventDef\n"
"%%EventDef PajeDefineEntityValue 5\n"
"%%  Alias         string\n"
"%%  EntityType    string\n"
"%%  Name          string\n"
"%%  Color         color\n"
"%%EndEventDef\n"
"%%EventDef PajeSetState 6\n"
"%%  Time          date\n"
"%%  Type          string\n"
"%%  Container     string\n"
"%%  Value         string\n"
"%%EndEventDef\n"
"1 W 0 Worker\n"
);
	printf("3 S W \"Worker State\"\n");
	for (t = 0; t < nt; t++)
		printf("5 R%d S Running_%d \"0.0 1.0 0.0\"\n", t, t);
	printf("5 F S Idle \"1.0 0.0 0.0\"\n");
	for (i = 0; i < nw; i++)
		printf("2 0 W%d W 0 \"%d\"\n", i, i);

	for (w = 0; w < nw; w++)
		printf("4 %f W%d W\n", tmax, w);

	fprintf(stderr,"%d workers, %d tasks\n", nw, nt);
	{
		struct task task[nt];
		memset(&task, 0, sizeof(task));
		for (t = nt-1; t >= 0; t--)
		{
			assert(scanf("c%d %lf\n", &foo, &task[t].stop) == 2);
		}

		for (t = nt-1; t >= 0; t--)
			for (w = 0; w < nw; w++)
			{
				assert(scanf("t%dw%d %lf\n", &tt, &ww, &bar) == 3);
				assert(ww == w);

				if (bar > 0.5)
				{
					task[t].num = tt;
					task[t].worker = w;
				}
		}
		for (t = nt-1; t >= 0; t--)
		{
			assert(scanf("s%d %lf\n", &tt, &task[t].start) == 2);
			fprintf(stderr,"%d: task %d on %d: %f - %f\n", nt-1-t, tt, task[t].worker, task[t].start, task[t].stop);
			assert(tt == task[t].num);
		}

		for (t = 0; t < nt; t++)
		{
			printf("6 %f S W%d R%d\n", task[t].start, task[t].worker, t);
			printf("6 %f S W%d F\n", task[t].stop, task[t].worker);
		}

		for (t = 0; t < nt; t++)
		{
			int t2;
			for (t2 = 0; t2 < nt; t2++)
			{
				if (t != t2 && task[t].worker == task[t2].worker)
				{
					if (!(task[t].start >= task[t2].stop
					    || task[t2].start >= task[t].stop))
					{
						fprintf(stderr,"oops, %d and %d sharing worker %d !!\n", task[t].num, task[t2].num, task[t].worker);
					}
				}
			}
		}
	}

	return 0;
}
