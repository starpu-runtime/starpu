/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2010-2015  Université de Bordeaux
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

void _starpu_fxt_write_paje_header(FILE *file STARPU_ATTRIBUTE_UNUSED)
{
	unsigned i;
#ifdef STARPU_HAVE_POTI
	poti_header(1, 1); /* 1 as parameter means basic, no extended events */
#else
	fprintf(file, "%%EventDef	PajeDefineContainerType	1\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDefineEventType	2\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDefineStateType	3\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDefineVariableType	4\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDefineLinkType	5\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	StartContainerType	string\n");
	fprintf(file, "%%	EndContainerType	string\n");
	fprintf(file, "%%	Name	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeDefineEntityValue	6\n");
	fprintf(file, "%%	Alias	string\n");
	fprintf(file, "%%	Type	string\n");
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
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Value	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajePopState	12\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Type	string\n");
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
	fprintf(file, "%%	StartContainer	string\n");
	fprintf(file, "%%	Key	string\n");
	fprintf(file, "%%EndEventDef\n");
	fprintf(file, "%%EventDef	PajeEndLink	19\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Value	string\n");
	fprintf(file, "%%	EndContainer	string\n");
	fprintf(file, "%%	Key	string\n");
	fprintf(file, "%%EndEventDef\n");
#ifdef STARPU_ENABLE_PAJE_CODELET_DETAILS
	fprintf(file, "%%EventDef PajeSetState 20\n");
	fprintf(file, "%%	Time	date\n");
	fprintf(file, "%%	Container	string\n");
	fprintf(file, "%%	Type	string\n");
	fprintf(file, "%%	Value	string\n");
	fprintf(file, "%%	Size	string\n");
	fprintf(file, "%%	Params	string\n");
	fprintf(file, "%%	Footprint	string\n");
	fprintf(file, "%%	Tag	string\n");
	fprintf(file, "%%	JobId	string\n");
	fprintf(file, "%%EndEventDef\n");
#endif
#endif

#ifdef STARPU_HAVE_POTI
	poti_DefineContainerType("MPIP", "0", "MPI Program");
	poti_DefineContainerType("P", "MPIP", "Program");
	poti_DefineContainerType("Mn", "P", "Memory Node");
	poti_DefineContainerType("T", "Mn", "Thread");
	poti_DefineContainerType("Mm", "Mn", "Memory Manager");
	poti_DefineContainerType("W", "T", "Worker");
	poti_DefineContainerType("MPICt", "T", "MPI Communication Thread");
	poti_DefineContainerType("Sc", "P", "Scheduler");
	poti_DefineEventType("prog_event", "P", "program event type");

	/* Types for the memory node */
	poti_DefineVariableType("bw", "Mm", "Bandwidth", "0 0 0");
	poti_DefineStateType("MS", "Mm", "Memory Node State");
	poti_DefineEntityValue("A", "MS", "Allocating", ".4 .1 .0");
	poti_DefineEntityValue("Ar", "MS", "AllocatingReuse", ".1 .1 .8");
	poti_DefineEntityValue("F", "MS", "Freeing", ".6 .3 .0");
	poti_DefineEntityValue("W", "MS", "WritingBack", ".0 .0 .5");
	poti_DefineEntityValue("Wa", "MS", "WritingBackAsync", ".0 .0 .4");
	poti_DefineEntityValue("R", "MS", "Reclaiming", ".0 .1 .6");
	poti_DefineEntityValue("Co", "MS", "DriverCopy", ".3 .5 .1");
	poti_DefineEntityValue("CoA", "MS", "DriverCopyAsync", ".1 .3 .1");
	poti_DefineEntityValue("No", "MS", "Nothing", ".0 .0 .0");

	/* Types for the Worker of the Memory Node */
	poti_DefineEventType("user_event", "T", "user event type");
	poti_DefineEventType("thread_event", "T", "thread event type");
	poti_DefineStateType("S", "T", "Thread State");
	poti_DefineEntityValue("I", "S", "Idle", ".9 .1 0");
	poti_DefineEntityValue("In", "S", "Initializing", "0.0 .7 1.0");
	poti_DefineEntityValue("D", "S", "Deinitializing", "0.0 .1 .7");
	poti_DefineEntityValue("Fi", "S", "FetchingInput", "1.0 .1 1.0");
	poti_DefineEntityValue("Po", "S", "PushingOutput", "0.1 1.0 1.0");
	poti_DefineEntityValue("C", "S", "Callback", ".0 .3 .8");
	poti_DefineEntityValue("B", "S", "Overhead", ".5 .18 .0");
	poti_DefineEntityValue("E", "S", "Executing", ".0 .6 .5");
	poti_DefineEntityValue("Sc", "S", "Scheduling", ".7 .36 .0");
	poti_DefineEntityValue("Sl", "S", "Sleeping", ".9 .1 .0");
	poti_DefineEntityValue("P", "S", "Progressing", ".4 .1 .6");
	poti_DefineEntityValue("U", "S", "Unpartitioning", ".0 .0 1.0");
	poti_DefineEntityValue("H", "S", "Hypervisor", ".5 .18 .0");
	poti_DefineStateType("WS", "W", "Worker State");
	poti_DefineEntityValue("I", "WS", "Idle", ".9 .1 .0");
	poti_DefineEntityValue("In", "WS", "Initializing", "0.0 .7 1.0");
	poti_DefineEntityValue("D", "WS", "Deinitializing", "0.0 .1 .7");
	poti_DefineEntityValue("Fi", "WS", "FetchingInput", "1.0 .1 1.0");
	poti_DefineEntityValue("Po", "WS", "PushingOutput", "0.1 1.0 1.0");
	poti_DefineEntityValue("C", "WS", "Callback", ".0 .3 .8");
	poti_DefineEntityValue("B", "WS", "Overhead", ".5 .18 .0");
	poti_DefineEntityValue("E", "WS", "Executing", ".0 .6 .5");
	poti_DefineEntityValue("Sc", "WS", "Scheduling", ".7 .36 .0");
	poti_DefineEntityValue("Sl", "WS", "Sleeping", ".9 .1 .0");
	poti_DefineEntityValue("P", "WS", "Progressing", ".4 .1 .6");
	poti_DefineEntityValue("U", "WS", "Unpartitioning", ".0 .0 1.0");
	poti_DefineEntityValue("H", "WS", "Hypervisor", ".5 .18 .0");

	/* Types for the MPI Communication Thread of the Memory Node */
	poti_DefineEventType("MPIev", "MPICt", "MPI event type");
	poti_DefineStateType("CtS", "MPICt", "Communication Thread State");
	poti_DefineEntityValue("P", "CtS", "Processing", "0 0 0");
	poti_DefineEntityValue("Sl", "CtS", "Sleeping", ".9 .1 .0");
	poti_DefineEntityValue("UT", "CtS", "UserTesting", ".2 .1 .6");
	poti_DefineEntityValue("UW", "CtS", "UserWaiting", ".4 .1 .3");
	poti_DefineEntityValue("SdS", "CtS", "SendSubmitted", "1.0 .1 1.0");
	poti_DefineEntityValue("RvS", "CtS", "RecieveSubmitted", "0.1 1.0 1.0");
	poti_DefineEntityValue("SdC", "CtS", "SendCompleted", "1.0 .5 1.0");
	poti_DefineEntityValue("RvC", "CtS", "RecieveCompleted", "0.5 1.0 1.0");

	for (i=1; i<STARPU_NMAX_SCHED_CTXS; i++)
	{
		char inctx[10];
		snprintf(inctx, sizeof(inctx), "InCtx%u", i);
		char *ctx = inctx+2;
		poti_DefineStateType(ctx, "T", inctx);
		poti_DefineEntityValue("I", ctx, "Idle", ".9 .1 .0");
		poti_DefineEntityValue("In", ctx, "Initializing", "0.0 .7 1.0");
		poti_DefineEntityValue("D", ctx, "Deinitializing", "0.0 .1 .7");
		poti_DefineEntityValue("Fi", ctx, "FetchingInput", "1.0 .1 1.0");
		poti_DefineEntityValue("Po", ctx, "PushingOutput", "0.1 1.0 1.0");
		poti_DefineEntityValue("C", ctx, "Callback", ".0 .3 .8");
		poti_DefineEntityValue("B", ctx, "Overhead", ".5 .18 .0");
		poti_DefineEntityValue("E", ctx, "Executing", ".0 .6 .5");
		poti_DefineEntityValue("Sc", ctx, "Scheduling", ".7 .36 .0");
		poti_DefineEntityValue("Sl", ctx, "Sleeping", ".9 .1 .0");
		poti_DefineEntityValue("P", ctx, "Progressing", ".4 .1 .6");
		poti_DefineEntityValue("U", ctx, "Unpartitioning", ".0 .0 1.0");
		poti_DefineEntityValue("H", ctx, "Hypervisor", ".5 .18 .0");
	}

	/* Types for the Scheduler */
	poti_DefineVariableType("nsubmitted", "Sc", "Number of Submitted Uncompleted Tasks", "0 0 0");
	poti_DefineVariableType("nready", "Sc", "Number of Ready Tasks", "0 0 0");

	/* Link types */
	poti_DefineLinkType("MPIL", "P", "MPICt", "MPICt", "MPI communications");
	poti_DefineLinkType("F", "P", "Mm", "Mm", "Intra-node data Fetch");
	poti_DefineLinkType("PF", "P", "Mm", "Mm", "Intra-node data PreFetch");
	poti_DefineLinkType("IF", "P", "Mm", "Mm", "Intra-node data IdleFetch");
	poti_DefineLinkType("WSL", "P", "W", "W", "Work steals");

	/* Creating the MPI Program */
	poti_CreateContainer(0, "MPIroot", "MPIP", "0", "root");
#else
	fprintf(file, "                                        \n\
1       MPIP      0       \"MPI Program\"                      	\n\
1       P      MPIP       \"Program\"                      	\n\
1       Mn      P       \"Memory Node\"                         \n\
1       T      Mn       \"Thread\"                               \n\
1       Mm      Mn       \"Memory Manager\"                         \n\
1       W      T       \"Worker\"                               \n\
1       MPICt   T       \"MPI Communication Thread\"              \n\
1       Sc       P       \"Scheduler State\"                        \n\
2       prog_event   P       \"program event type\"				\n\
2       user_event   T       \"user event type\"				\n\
2       thread_event   T       \"thread event type\"				\n\
2       MPIev   MPICt    \"MPI event type\"			\n\
3       S       T       \"Thread State\"                        \n\
3       CtS     MPICt    \"Communication Thread State\"          \n");
	for (i=1; i<STARPU_NMAX_SCHED_CTXS; i++)
		fprintf(file, "3       Ctx%u      T     \"InCtx%u\"         		\n", i, i);
	fprintf(file, "\
3       MS       Mm       \"Memory Node State\"                        \n\
4       nsubmitted    Sc       \"Number of Submitted Uncompleted Tasks\"                        \n\
4       nready    Sc       \"Number of Ready Tasks\"                        \n\
4       bw      Mm       \"Bandwidth\"                        \n\
6       I       S       Idle         \".9 .1 .0\"		\n\
6       In       S      Initializing       \"0.0 .7 1.0\"            \n\
6       D       S      Deinitializing       \"0.0 .1 .7\"            \n\
6       Fi       S      FetchingInput       \"1.0 .1 1.0\"            \n\
6       Po       S      PushingOutput       \"0.1 1.0 1.0\"            \n\
6       C       S       Callback       \".0 .3 .8\"            \n\
6       B       S       Overhead         \".5 .18 .0\"		\n\
6       E       S       Executing         \".0 .6 .5\"		\n\
6       Sc       S      Scheduling         \".7 .36 .0\"		\n\
6       Sl       S      Sleeping         \".9 .1 .0\"		\n\
6       P       S       Progressing         \".4 .1 .6\"		\n\
6       U       S       Unpartitioning      \".0 .0 1.0\"		\n\
6       H       S       Hypervisor      \".5 .18 .0\"		\n\
3       WS       W       \"Worker State\"                        \n\
6       I       WS       Idle         \".9 .1 .0\"		\n\
6       In       WS      Initializing       \"0.0 .7 1.0\"            \n\
6       D       WS      Deinitializing       \"0.0 .1 .7\"            \n\
6       Fi       WS      FetchingInput       \"1.0 .1 1.0\"            \n\
6       Po       WS      PushingOutput       \"0.1 1.0 1.0\"            \n\
6       C       WS       Callback       \".0 .3 .8\"            \n\
6       B       WS       Overhead         \".5 .18 .0\"		\n\
6       E       WS       Executing         \".0 .6 .5\"		\n\
6       Sc       WS      Scheduling         \".7 .36 .0\"		\n\
6       Sl       WS      Sleeping         \".9 .1 .0\"		\n\
6       P       WS       Progressing         \".4 .1 .6\"		\n\
6       U       WS       Unpartitioning      \".0 .0 1.0\"		\n\
6       H       WS       Hypervisor      \".5 .18 .0\"		\n");
	fprintf(file, "\
6       P       CtS       Processing         \"0 0 0\"		\n\
6       Sl       CtS      Sleeping         \".9 .1 .0\"		\n\
6       UT       CtS      UserTesting        \".2 .1 .6\"	\n\
6       UW       CtS      UserWaiting        \".4 .1 .3\"	\n\
6       SdS       CtS      SendSubmitted     \"1.0 .1 1.0\"	\n\
6       RvS       CtS      RecieveSubmitted  \"0.1 1.0 1.0\"	\n\
6       SdC       CtS      SendCompleted     \"1.0 .5 1.0\"	\n\
6       RvC       CtS      RecieveCompleted  \"0.5 1.0 1.0\"	\n");
	for (i=1; i<STARPU_NMAX_SCHED_CTXS; i++)
		fprintf(file, "\
6       I       Ctx%u      Idle         \".9 .1 .0\"		\n\
6       In       Ctx%u      Initializing       \"0.0 .7 1.0\"            \n\
6       D       Ctx%u      Deinitializing       \"0.0 .1 .7\"            \n\
6       Fi       Ctx%u      FetchingInput       \"1.0 .1 1.0\"            \n\
6       Po       Ctx%u      PushingOutput       \"0.1 1.0 1.0\"            \n\
6       C       Ctx%u       Callback       \".0 .3 .8\"            \n\
6       B       Ctx%u       Overhead         \".5 .18 .0\"		\n\
6       E       Ctx%u       Executing         \".0 .6 .5\"		\n\
6       Sc       Ctx%u      Scheduling         \".7 .36 .0\"		\n\
6       Sl       Ctx%u      Sleeping         \".9 .1 .0\"		\n\
6       P       Ctx%u       Progressing         \".4 .1 .6\"		\n\
6       U       Ctx%u       Unpartitioning         \".0 .0 1.0\"		\n",
		i, i, i, i, i, i, i, i, i, i, i, i);
	fprintf(file, "\
6       A       MS      Allocating         \".4 .1 .0\"		\n\
6       Ar       MS      AllocatingReuse       \".1 .1 .8\"		\n\
6       F       MS      Freeing         \".6 .3 .0\"		\n\
6       W       MS      WritingBack         \".0 .0 .5\"		\n\
6       Wa       MS     WritingBackAsync    \".0 .0 .4\"		\n\
6       R       MS      Reclaiming         \".0 .1 .6\"		\n\
6       Co       MS     DriverCopy         \".3 .5 .1\"		\n\
6       CoA      MS     DriverCopyAsync         \".1 .3 .1\"		\n\
6       No       MS     Nothing         \".0 .0 .0\"		\n\
5       MPIL     P	MPICt	MPICt   MPIL			\n\
5       F       P	Mm	Mm      F\n\
5       PF      P	Mm	Mm      PF\n\
5       IF      P	Mm	Mm      IF\n\
5       WSL     P	W	W       WSL\n");

	fprintf(file, "7      0.0 MPIroot      MPIP      0       root\n");
#endif
}

#endif
