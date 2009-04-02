/*
 * StarPU
 * Copyright (C) INRIA 2008-2009 (see AUTHORS file)
 *
 * This program is free software; you can redistribute it and/or modify
 * it under the terms of the GNU Lesser General Public License as published by
 * the Free Software Foundation; either version 2.1 of the License, or (at
 * your option) any later version.
 *
 * This program is distributed in the hope that it will be useful, but
 * WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 *
 * See the GNU Lesser General Public License in COPYING.LGPL for more details.
 */

#include "histo-paje.h"

void write_paje_header(FILE *file)
{
	fprintf(file, "\%%EventDef	PajeDefineContainerType	1\n");
	fprintf(file, "\%%	Alias	string\n");
	fprintf(file, "\%%	ContainerType	string\n");
	fprintf(file, "\%%	Name	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeDefineEventType	2\n");
	fprintf(file, "\%%	Alias	string\n");
	fprintf(file, "\%%	ContainerType	string\n");
	fprintf(file, "\%%	Name	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeDefineStateType	3\n");
	fprintf(file, "\%%	Alias	string\n");
	fprintf(file, "\%%	ContainerType	string\n");
	fprintf(file, "\%%	Name	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeDefineVariableType	4\n");
	fprintf(file, "\%%	Alias	string\n");
	fprintf(file, "\%%	ContainerType	string\n");
	fprintf(file, "\%%	Name	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeDefineLinkType	5\n");
	fprintf(file, "\%%	Alias	string\n");
	fprintf(file, "\%%	ContainerType	string\n");
	fprintf(file, "\%%	SourceContainerType	string\n");
	fprintf(file, "\%%	DestContainerType	string\n");
	fprintf(file, "\%%	Name	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeDefineEntityValue	6\n");
	fprintf(file, "\%%	Alias	string\n");
	fprintf(file, "\%%	EntityType	string\n");
	fprintf(file, "\%%	Name	string\n");
	fprintf(file, "\%%	Color	color\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeCreateContainer	7\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Alias	string\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Name	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeDestroyContainer	8\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Name	string\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeNewEvent	9\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef PajeSetState 10\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajePushState	11\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajePushState	111\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	string\n");
	fprintf(file, "\%%	Object	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajePopState	12\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeSetVariable	13\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	double\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeAddVariable	14\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	double\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeSubVariable	15\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	double\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeStartLink	16\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	string\n");
	fprintf(file, "\%%	SourceContainer	string\n");
	fprintf(file, "\%%	Key	string\n");
	fprintf(file, "\%%	Size	int\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeEndLink	17\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	string\n");
	fprintf(file, "\%%	DestContainer	string\n");
	fprintf(file, "\%%	Key	string\n");
	fprintf(file, "\%%	Size	int\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeStartLink	18\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	string\n");
	fprintf(file, "\%%	SourceContainer	string\n");
	fprintf(file, "\%%	Key	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeEndLink	19\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	string\n");
	fprintf(file, "\%%	DestContainer	string\n");
	fprintf(file, "\%%	Key	string\n");
	fprintf(file, "\%%EndEventDef\n");
	fprintf(file, "\%%EventDef	PajeNewEvent   112\n");
	fprintf(file, "\%%	Time	date\n");
	fprintf(file, "\%%	Type	string\n");
	fprintf(file, "\%%	Container	string\n");
	fprintf(file, "\%%	Value	string\n");
	fprintf(file, "\%%       ThreadName      string\n");
	fprintf(file, "\%%       ThreadGroup     string\n");
	fprintf(file, "\%%       ThreadParent    string\n");
	fprintf(file, "\%%EndEventDef\n");

}
