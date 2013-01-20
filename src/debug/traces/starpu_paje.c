/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2013  Université de Bordeaux 1
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

#include "starpu_fxt.h"
#include <common/config.h>
#ifdef STARPU_HAVE_POTI
#include <poti.h>
#endif

#ifdef STARPU_USE_FXT

void _starpu_fxt_write_paje_header(FILE *file)
{
	unsigned i;
#ifdef STARPU_HAVE_POTI
	poti_header(1, 1); /* 1 as parameter means basic, no extended events */
#else
	fprintf(file, "%%EventDef	PajeDefineContainerType	1\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	ContainerType	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDefineEventType	2\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	ContainerType	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDefineStateType	3\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	ContainerType	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDefineVariableType	4\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	ContainerType	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDefineLinkType	5\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	ContainerType	string\n");
	fprintf(file, "%%	SourceContainerType	string\n");
	fprintf(file, "%%	DestContainerType	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDefineEntityValue	6\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	EntityType	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%	Color	color\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeCreateContainer	7\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDestroyContainer	8\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeNewEvent	9\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Value	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef PajeSetState 10\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Value	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajePushState	11\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Value	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajePopState	12\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeSetVariable	13\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Value	double\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeAddVariable	14\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Value	double\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeSubVariable	15\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Value	double\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeStartLink	18\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Value	string\n");
	fprintf(file, "%%	SourceContainer	string\n");
	fprintf(file, "%%	Key	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeEndLink	19\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Value	string\n");
	fprintf(file, "%%	DestContainer	string\n");
	fprintf(file, "%%	Key	string\n");
	fprintf(file, "%%EndEventDef\n");
#endif

#ifdef STARPU_HAVE_POTI
	poti_DefineContainerType("MPIP", "0", "MPI Program");
	poti_DefineContainerType("P", "MPIP", "Program");
	poti_DefineContainerType("Mn", "P", "Memory Node");
	poti_DefineContainerType("T", "Mn", "Thread");
	poti_DefineContainerType("W", "T", "Worker");
	poti_DefineContainerType("Sc", "P", "Scheduler");

	/* Types for the memory node */
	poti_DefineVariableType("bw", "Mn", "Bandwidth", "0 0 0");
	poti_DefineStateType("MS", "Mn", "Memory Node State");
	poti_DefineEntityValue("A", "MS", "Allocating", ".4 .1 .0");
	poti_DefineEntityValue("Ar", "MS", "AllocatingReuse", ".1 .1 .8");
	poti_DefineEntityValue("R", "MS", "Reclaiming", ".0 .1 .4");
	poti_DefineEntityValue("Co", "MS", "DriverCopy", ".3 .5 .1");
	poti_DefineEntityValue("CoA", "MS", "DriverCopyAsync", ".1 .3 .1");
	poti_DefineEntityValue("No", "MS", "Nothing", ".0 .0 .0");

	/* Types for the Worker of the Memory Node */
	poti_DefineEventType("event", "T", "event type");
	poti_DefineStateType("S", "T", "Thread State");
	poti_DefineEntityValue("I", "S", "Initializing", "0.0 .7 1.0");
	poti_DefineEntityValue("D", "S", "Deinitializing", "0.0 .1 .7");
	poti_DefineEntityValue("Fi", "S", "FetchingInput", "1.0 .1 1.0");
	poti_DefineEntityValue("Po", "S", "PushingOutput", "0.1 1.0 1.0");
	poti_DefineEntityValue("C", "S", "Callback", ".0 .3 .8");
	poti_DefineEntityValue("B", "S", "Blocked", ".9 .1 .0");
	poti_DefineEntityValue("Sl", "S", "Sleeping", ".9 .1 .0");
	poti_DefineEntityValue("P", "S", "Progressing", ".4 .1 .6");

	for (i=1; i<=10; i++)
	{
		char inctx[8];
		snprintf(inctx, sizeof(inctx), "InCtx%u", i);
		char *ctx = inctx+2;
		poti_DefineStateType(ctx, "T", inctx);
		poti_DefineEntityValue("I", ctx, "Initializing", "0.0 .7 1.0");
		poti_DefineEntityValue("D", ctx, "Deinitializing", "0.0 .1 .7");
		poti_DefineEntityValue("Fi", ctx, "FetchingInput", "1.0 .1 1.0");
		poti_DefineEntityValue("Po", ctx, "PushingOutput", "0.1 1.0 1.0");
		poti_DefineEntityValue("C", ctx, "Callback", ".0 .3 .8");
		poti_DefineEntityValue("B", ctx, "Blocked", ".9 .1 .0");
		poti_DefineEntityValue("Sl", ctx, "Sleeping", ".9 .1 .0");
		poti_DefineEntityValue("P", ctx, "Progressing", ".4 .1 .6");
	}

	/* Types for the Scheduler */
	poti_DefineVariableType("ntask", "Sc", "Number of tasks", "0 0 0");

	/* Link types */
	poti_DefineLinkType("MPIL", "MPIP", "P", "P", "Links between two MPI programs");
	poti_DefineLinkType("L", "P", "Mn", "Mn", "Links between two Memory Nodes");

	/* Creating the MPI Program */
	poti_CreateContainer(0, "MPIroot", "MPIP", "0", "root");
#else
	fprintf(file, "                                        \n\
1       MPIP      0       \"MPI Program\"                      	\n\
1       P      MPIP       \"Program\"                      	\n\
1       Mn      P       \"Memory Node\"                         \n\
1       T      Mn       \"Thread\"                               \n\
1       W      T       \"Worker\"                               \n\
1       Sc       P       \"Scheduler State\"                        \n\
2       event   T       \"event type\"				\n\
3       S       T       \"Thread State\"                        \n");
	for (i=1; i<=10; i++)
		fprintf(file, "3       Ctx%u      T     \"InCtx%u\"         		\n", i, i);
	fprintf(file, "\
3       MS       Mn       \"Memory Node State\"                        \n\
4       ntask    Sc       \"Number of tasks\"                        \n\
4       bw      Mn       \"Bandwidth\"                        \n\
6       I       S      Initializing       \"0.0 .7 1.0\"            \n\
6       D       S      Deinitializing       \"0.0 .1 .7\"            \n\
6       Fi       S      FetchingInput       \"1.0 .1 1.0\"            \n\
6       Po       S      PushingOutput       \"0.1 1.0 1.0\"            \n\
6       C       S       Callback       \".0 .3 .8\"            \n\
6       B       S       Blocked         \".9 .1 .0\"		\n\
6       Sl       S      Sleeping         \".9 .1 .0\"		\n\
6       P       S       Progressing         \".4 .1 .6\"		\n");
	for (i=1; i<=10; i++)
		fprintf(file, "\
6       I       Ctx%u      Initializing       \"0.0 .7 1.0\"            \n\
6       D       Ctx%u      Deinitializing       \"0.0 .1 .7\"            \n\
6       Fi       Ctx%u      FetchingInput       \"1.0 .1 1.0\"            \n\
6       Po       Ctx%u      PushingOutput       \"0.1 1.0 1.0\"            \n\
6       C       Ctx%u       Callback       \".0 .3 .8\"            \n\
6       B       Ctx%u       Blocked         \".9 .1 .0\"		\n\
6       Sl       Ctx%u      Sleeping         \".9 .1 .0\"		\n\
6       P       Ctx%u       Progressing         \".4 .1 .6\"		\n",
		i, i, i, i, i, i, i, i);
	fprintf(file, "\
6       A       MS      Allocating         \".4 .1 .0\"		\n\
6       Ar       MS      AllocatingReuse       \".1 .1 .8\"		\n\
6       R       MS      Reclaiming         \".0 .1 .4\"		\n\
6       Co       MS     DriverCopy         \".3 .5 .1\"		\n\
6       CoA      MS     DriverCopyAsync         \".1 .3 .1\"		\n\
6       No       MS     Nothing         \".0 .0 .0\"		\n\
5       MPIL     MPIP	P	P      MPIL\n\
5       L       P	Mn	Mn      L\n");

	fprintf(file, "7      0.0 MPIroot      MPIP      0       root\n");
#endif
}

#endif
