/* StarPU --- Runtime system for heterogeneous multicore architectures.
 *
 * Copyright (C) 2015-2020  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
 * Copyright (C) 2015       Anthony Simonet
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

#include <stdint.h>
#include <stdlib.h>
#include <common/uthash.h>
#include <starpu.h>
#include "starpu_fxt.h"

#ifdef STARPU_USE_FXT
static struct component
{
	UT_hash_handle hh;
	char *name;
	int workerid;
	uint64_t ptr;
	unsigned nchildren;
	struct component **children;
	struct component *parent;
	unsigned ntasks;
	unsigned npriotasks;
} *components;

static unsigned global_state = 1;
static unsigned nsubmitted;
static unsigned curq_size;
static unsigned nflowing;

#define COMPONENT_ADD(head, field, add) HASH_ADD(hh, head, field, sizeof(uint64_t), add);
#define COMPONENT_FIND(head, find, out) HASH_FIND(hh, head, &find, sizeof(uint64_t), out);

static struct component *fxt_component_root(void)
{
	struct component *comp=NULL, *tmp=NULL;
	HASH_ITER(hh, components, comp, tmp)
	{
		while (comp->parent)
			comp = comp->parent;
		return comp;
	}
	return NULL;
}

void _starpu_fxt_component_new(uint64_t component, char *name)
{
	struct component *comp;
	_STARPU_MALLOC(comp, sizeof(*comp));

	if (!strncmp(name, "worker ", 7))
	{
		comp->name = strdup("worker");
		comp->workerid = atoi(name+7);
	}
	else
	{
		comp->name = strdup(name);
		comp->workerid = -1;
	}
	comp->ptr = component;
	comp->nchildren = 0;
	comp->children = NULL;
	comp->parent = NULL;
	comp->ntasks = 0;
	comp->npriotasks = 0;

	COMPONENT_ADD(components, ptr, comp);
}

static void fxt_component_dump(FILE *file, struct component *comp, unsigned depth)
{
	unsigned i;
	fprintf(file,"%*s%s (%d %"PRIx64", %d tasks %d prio tasks)\n", 2*depth, "", comp->name, depth, comp->ptr, comp->ntasks, comp->npriotasks);
	for (i = 0; i < comp->nchildren; i++)
		if (comp->children[i]->parent == comp)
			fxt_component_dump(file, comp->children[i], depth+1);
}

void _starpu_fxt_component_dump(FILE *file)
{
	fxt_component_dump(file, fxt_component_root(), 0);
}

static void fxt_worker_print(FILE *file, struct starpu_fxt_options *options, int workerid, unsigned comp_workerid, unsigned depth)
{
	fprintf(file, "\t\t\t%*s<table><tr><td class='worker_box%s'><center>%s\n", 2*depth, "",
		(int) comp_workerid == workerid ? "_sched":"",
		options->worker_names[comp_workerid]);
	if (_starpu_last_codelet_symbol[comp_workerid][0])
		fprintf(file, "\t\t\t%*s<table><tr><td class='run_task'>%s</td></tr></table>\n", 2*(depth+1), "", _starpu_last_codelet_symbol[comp_workerid]);
	else
		fprintf(file, "\t\t\t%*s<table><tr><td class='fake_task'></td></tr></table>\n", 2*(depth+1), "");
	fprintf(file, "\t\t\t%*s</center></td></tr>\n", 2*depth, "");
	fprintf(file, "\t\t\t%*s</table>", 2*depth, "");
}

static void fxt_component_print(FILE *file, struct starpu_fxt_options *options, int workerid, struct component *from, struct component *to, struct component *comp, unsigned depth)
{
	unsigned i, n;
	unsigned ntasks = comp->ntasks + comp->npriotasks;

	if (from == comp)
		/* Additionally show now-empty slot */
		ntasks++;

	for (i = 0, n = 0; i < comp->nchildren; i++)
		if (comp->children[i]->parent == comp)
			n++;
	fprintf(file, "\t\t\t%*s<table><tr><td class='box' colspan=%u><center>%s\n", 2*depth, "", n, comp->name);

	if (!strcmp(comp->name,"prio") || !strcmp(comp->name,"fifo") || !strcmp(comp->name,"heft") || !strcmp(comp->name,"work_stealing"))
	{
		/* Show task queue */
#define N 3
		n = ntasks;
		if (n > N)
			n = N;
		for (i = 0; i < N-n; i++)
			fprintf(file, "\t\t\t%*s<table><tr><td class='fake_task'></td></tr></table>\n", 2*depth, "");
		if (ntasks)
		{
			if (ntasks > N)
				fprintf(file, "\t\t\t%*s<table><tr><td class='%s'>%u</td></tr></table>\n", 2*depth, "",
					from == comp
						? (comp->npriotasks >= N ? "last_task_full_prio" : "last_task_full")
						: (comp->npriotasks >= N ? "task_prio" : "task"),
					comp->ntasks + comp->npriotasks);
			else
				fprintf(file, "\t\t\t%*s<table><tr><td class='%s'></td></tr></table>\n", 2*depth, "",
					from == comp
						? "last_task_empty"
						: (comp->ntasks ? "task" : "task_prio"));
			for (i = 1; i < n; i++)
				fprintf(file, "\t\t\t%*s<table><tr><td class='%s'></td></tr></table>\n", 2*depth, "",
					n - i > comp->npriotasks ? "task" : "task_prio");
		}
	}
	else
	{
		if (ntasks == 0)
			fprintf(file, "\t\t\t%*s<table><tr><td class='fake_task'></td></tr></table>\n", 2*depth, "");
		else if (ntasks == 1)
			fprintf(file, "\t\t\t%*s<table><tr><td class='%s'></td></tr></table>\n", 2*depth, "",
				from == comp
					? "last_task_empty"
					: (comp->npriotasks ? "task_prio" : "task"));
		else
			fprintf(file, "\t\t\t%*s<table><tr><td class='%s'>%u</td></tr></table>\n", 2*depth, "",
				from == comp
					? (comp->npriotasks ? "last_task_full_prio" : "last_task_full")
					: (comp->npriotasks ? "task_prio" : "task"), comp->ntasks + comp->npriotasks);
	}
	fprintf(file, "\t\t\t%*s</center></td></tr>\n", 2*depth, "");

	if (comp->nchildren > 0)
	{
		fprintf(file, "\t\t\t%*s<tr>\n", 2*depth, "");
		for (i = 0; i < comp->nchildren; i++)
			if (comp->children[i]->parent == comp)
			{
				fprintf(file, "\t\t\t%*s<td>\n", 2*depth, "");
				fxt_component_print(file, options, workerid, from, to, comp->children[i], depth+1);
				fprintf(file, "\t\t\t%*s</td>\n", 2*depth, "");
			}
		fprintf(file, "\t\t\t%*s</tr>\n", 2*depth, "");
	}

	if (!strcmp(comp->name, "worker"))
	{
		fprintf(file, "\t\t\t%*s<tr>\n", 2*depth, "");
		fprintf(file, "\t\t\t%*s<td>\n", 2*depth, "");
		fxt_worker_print(file, options, workerid, comp->workerid, depth+1);
		fprintf(file, "\t\t\t%*s</td>\n", 2*depth, "");
		fprintf(file, "\t\t\t%*s</tr>\n", 2*depth, "");
	}

	fprintf(file, "\t\t\t%*s</table>", 2*depth, "");
}

void _starpu_fxt_component_print(FILE *file, struct starpu_fxt_options *options, int workerid, struct component *from, struct component *to)
{
	fprintf(file, "<center>\n");
	fxt_component_print(file, options, workerid, from, to, fxt_component_root(), 0);
	fprintf(file, "</center>\n");
}

void _starpu_fxt_component_print_header(FILE *file)
{
	/* CSS and Javascript code from Anthony Simonet */
	fprintf(file, "<!DOCTYPE html>\n");
	fprintf(file, "<html lang='fr'>\n");
	
	fprintf(file, "\t<head>\n");
	fprintf(file, "\t\t<meta charset='utf-8'>\n");
	fprintf(file, "\t\t<link rel='stylesheet' href='http://code.jquery.com/ui/1.11.2/themes/smoothness/jquery-ui.css'>\n");
	fprintf(file, "\t\t<script src='http://code.jquery.com/jquery-1.10.2.js'></script>\n");
	fprintf(file, "\t\t<script src='http://code.jquery.com/ui/1.11.2/jquery-ui.js'></script>\n");
	//fprintf(file, "\t\t<link rel='stylesheet' href='/resources/demos/style.css'>\n");
	//fprintf(file, "\t\t<link rel='stylesheet' type='text/css' href='../styles.css'>\n");

	fprintf(file, "\t\t<style>\n");

	fprintf(file, "\t\t\ttable {\n");
	fprintf(file, "\t\t\t\tmargin: 0;\n");
	fprintf(file, "\t\t\t\tpadding: 0;\n");
	fprintf(file, "\t\t\t}\n");

	fprintf(file, "\t\t\ttd {\n");
	fprintf(file, "\t\t\t\tmargin: 0;\n");
	fprintf(file, "\t\t\t\tpadding: 0;\n");
	fprintf(file, "\t\t\t\tvertical-align: top;\n");
	fprintf(file, "\t\t\t\ttext-align: center;\n");
	fprintf(file, "\t\t\t}\n");

	fprintf(file, "\t\t\ttd.box {\n");
	fprintf(file, "\t\t\t\tborder: solid 1px;\n");
	fprintf(file, "\t\t\t}\n");

	fprintf(file, "\t\t\ttd.worker_box {\n");
	fprintf(file, "\t\t\t\tborder: solid 1px;\n");
	/* Fixed width to make output more homogeneous */
	fprintf(file, "\t\t\t\twidth: 75px;\n");
	fprintf(file, "\t\t\t}\n");

	fprintf(file, "\t\t\ttd.worker_box_sched {\n");
	fprintf(file, "\t\t\t\tborder: solid 1px;\n");
	/* Fixed width to make output more homogeneous */
	fprintf(file, "\t\t\t\twidth: 75px;\n");
	fprintf(file, "\t\t\t\tbackground-color: lightgreen;\n");
	fprintf(file, "\t\t\t}\n");

	/* Task */
	fprintf(file, "\t\t\ttd.task {\n");
	fprintf(file, "\t\t\t\tborder: solid 1px;\n");
	fprintf(file, "\t\t\t\twidth: 23px;\n");
	fprintf(file, "\t\t\t\theight: 23px;\n");
	fprintf(file, "\t\t\t\tbackground-color: #87CEEB;\n");
	fprintf(file, "\t\t\t}\n");

	/* Task being run (with codelet name) */
	fprintf(file, "\t\t\ttd.run_task {\n");
	fprintf(file, "\t\t\t\tborder: solid 1px;\n");
	fprintf(file, "\t\t\t\twidth: 69px;\n");
	fprintf(file, "\t\t\t\tmax-width: 69px;\n");
	fprintf(file, "\t\t\t\toverflow: hidden;\n");
	fprintf(file, "\t\t\t\tfont-size: 50%%;\n");
	fprintf(file, "\t\t\t\theight: 23px;\n");
	fprintf(file, "\t\t\t\tbackground-color: #87CEEB;\n");
	fprintf(file, "\t\t\t}\n");

	/* Prioritized Task */
	fprintf(file, "\t\t\ttd.task_prio {\n");
	fprintf(file, "\t\t\t\tborder: solid 1px;\n");
	fprintf(file, "\t\t\t\twidth: 23px;\n");
	fprintf(file, "\t\t\t\theight: 23px;\n");
	fprintf(file, "\t\t\t\tbackground-color: red;\n");
	fprintf(file, "\t\t\t}\n");

	/* Slot of previous task */
	fprintf(file, "\t\t\ttd.last_task_empty {\n");
	fprintf(file, "\t\t\t\tborder: dashed 1px;\n");
	fprintf(file, "\t\t\t\twidth: 23px;\n");
	fprintf(file, "\t\t\t\theight: 23px;\n");
	fprintf(file, "\t\t\t\tbackground-color: white;\n");
	fprintf(file, "\t\t\t}\n");

	/* Slot of previous task (but still other tasks) */
	fprintf(file, "\t\t\ttd.last_task_full {\n");
	fprintf(file, "\t\t\t\tborder: dashed 1px;\n");
	fprintf(file, "\t\t\t\twidth: 23px;\n");
	fprintf(file, "\t\t\t\theight: 23px;\n");
	fprintf(file, "\t\t\t\tbackground-color: #87CEEB;\n");
	fprintf(file, "\t\t\t}\n");

	/* Slot of previous task (but still other prioritized) */
	fprintf(file, "\t\t\ttd.last_task_full_prio {\n");
	fprintf(file, "\t\t\t\tborder: dashed 1px;\n");
	fprintf(file, "\t\t\t\twidth: 23px;\n");
	fprintf(file, "\t\t\t\theight: 23px;\n");
	fprintf(file, "\t\t\t\tbackground-color: red;\n");
	fprintf(file, "\t\t\t}\n");

	/* Empty task slot */
	fprintf(file, "\t\t\ttd.fake_task {\n");
	fprintf(file, "\t\t\t\twidth: 25px;\n");
	fprintf(file, "\t\t\t\theight: 25px;\n");
	fprintf(file, "\t\t\t}\n");

	fprintf(file, "\t\t</style>\n");

	fprintf(file, "\t\t<script>\n");
	fprintf(file, "\t\t\tfunction getInput(){\n");
	fprintf(file, "\t\t\t\tvar input = document.getElementById('input').value;\n");
	fprintf(file, "\t\t\t\tif (input <= 0 || input > $('#slider').slider('option', 'max')){\n");
	fprintf(file, "\t\t\t\t\talert('Invalid state value');\n");
	fprintf(file, "\t\t\t\t}\n");
	fprintf(file, "\t\t\t\tdocument.getElementById('et' + document.getElementById('etape').value).style.display = 'none';\n");
	fprintf(file, "\t\t\t\tdocument.getElementById('et' + input).style.display = 'block';\n");
	fprintf(file, "\t\t\t\t$('#etape').val(input);\n");
	fprintf(file, "\t\t\t\t$('#slider').slider('value', input);\n");
	fprintf(file, "\t\t\t}\n");
	fprintf(file, "\t\t</script>\n");

	fprintf(file, "\t\t<script>\n");
	fprintf(file, "\t\t\tvar myVar = null;\n");
	fprintf(file, "\t\t\tfunction changeState(number){\n");
	fprintf(file, "\t\t\t\tvar state = document.getElementById('etape').value;\n");
	fprintf(file, "\t\t\t\tvar state2 = parseInt(state) + parseInt(number);\n");
	fprintf(file, "\t\t\t\tvar min = $('#slider').slider('option', 'min');\n");
	fprintf(file, "\t\t\t\tvar max = $('#slider').slider('option', 'max');\n");
	fprintf(file, "\t\t\t\t\tdocument.getElementById('et' + document.getElementById('etape').value).style.display = 'none';\n");
	fprintf(file, "\t\t\t\tif (state2 >= min && state2 <= max){\n");
	fprintf(file, "\t\t\t\t\tdocument.getElementById('et' + state2).style.display = 'block';\n");
	fprintf(file, "\t\t\t\t\t$('#etape').val(state2);\n");
	fprintf(file, "\t\t\t\t\t$('#slider').slider('value', state2);\n");
	fprintf(file, "\t\t\t\t}\n");
	fprintf(file, "\t\t\t\telse if (state2 < min){\n");
	fprintf(file, "\t\t\t\t\tdocument.getElementById('et' + min).style.display = 'block';\n");
	fprintf(file, "\t\t\t\t\t$('#etape').val(min);\n");
	fprintf(file, "\t\t\t\t\t$('#slider').slider('value', min);\n");
	fprintf(file, "\t\t\t\t}\n");
	fprintf(file, "\t\t\t\telse if (state2 > max){\n");
	fprintf(file, "\t\t\t\t\tdocument.getElementById('et' + max).style.display = 'block';\n");
	fprintf(file, "\t\t\t\t\t$('#etape').val(max);\n");
	fprintf(file, "\t\t\t\t\t$('#slider').slider('value', max);\n");
	fprintf(file, "\t\t\t\t}\n");
	fprintf(file, "\t\t\t}\n");
	fprintf(file, "\t\t</script>\n");

	fprintf(file, "\t</head>\n");

	fprintf(file, "\t<body>\n");
}

static void fxt_component_print_step(FILE *file, struct starpu_fxt_options *options, double timestamp, int workerid, unsigned push, struct component *from, struct component *to)
{
	fprintf(file, "\t\t<div id='et%u' style='display:%s;'><center><!-- Étape %u -->\n",
			global_state, global_state > 1 ? "none":"block", global_state);
	fprintf(file, "\t\t<p>Time %f, %u submitted %u ready, %s</p>\n", timestamp, nsubmitted, curq_size-nflowing, push?"push":"pull");
	//fprintf(file, "\t\t\t<tt><pre>\n");
	//_starpu_fxt_component_dump(file);
	//fprintf(file, "\t\t\t</pre></tt>\n");
	_starpu_fxt_component_print(file, options, workerid, from, to);
	fprintf(file,"\t\t</center></div>");

	global_state++;
}

void _starpu_fxt_component_connect(uint64_t parent, uint64_t child)
{
	struct component *parent_p, *child_p;
	unsigned n;

	COMPONENT_FIND(components, parent, parent_p);
	COMPONENT_FIND(components, child, child_p);
	STARPU_ASSERT(parent_p);
	STARPU_ASSERT(child_p);

	n = ++parent_p->nchildren;
	_STARPU_REALLOC(parent_p->children, n * sizeof(*parent_p->children));
	parent_p->children[n-1] = child_p;
	if (!child_p->parent)
		child_p->parent = parent_p;
}

void _starpu_fxt_component_update_ntasks(unsigned _nsubmitted, unsigned _curq_size)
{
	nsubmitted = _nsubmitted;
	curq_size = _curq_size;
}

void _starpu_fxt_component_push(FILE *output, struct starpu_fxt_options *options, double timestamp, int workerid, uint64_t from, uint64_t to, uint64_t task STARPU_ATTRIBUTE_UNUSED, unsigned prio)
{
	struct component *from_p = NULL, *to_p = NULL;

	if (to == from)
		return;

	if (from)
	{
		COMPONENT_FIND(components, from, from_p);
		STARPU_ASSERT(from_p);
	}
	if (to)
	{
		COMPONENT_FIND(components, to, to_p);
		STARPU_ASSERT(to_p);
	}
	if (from_p)
	{
		if (prio)
			from_p->npriotasks--;
		else
			from_p->ntasks--;
	}
	else
		nflowing++;
	if (to_p)
	{
		if (prio)
			to_p->npriotasks++;
		else
			to_p->ntasks++;
	}

	// fprintf(stderr,"push from %s to %s\n", from_p?from_p->name:"none", to_p?to_p->name:"none");
	fxt_component_print_step(output, options, timestamp, workerid, 1, from_p, to_p);
}

void _starpu_fxt_component_pull(FILE *output, struct starpu_fxt_options *options, double timestamp, int workerid, uint64_t from, uint64_t to, uint64_t task STARPU_ATTRIBUTE_UNUSED, unsigned prio)
{
	struct component *from_p = NULL, *to_p = NULL;

	if (to == from)
		return;

	if (from)
	{
		COMPONENT_FIND(components, from, from_p);
		STARPU_ASSERT(from_p);
	}
	if (to)
	{
		COMPONENT_FIND(components, to, to_p);
		STARPU_ASSERT(to_p);
	}
	if (from_p)
	{
		if (prio)
			from_p->npriotasks--;
		else
			from_p->ntasks--;
	}
	if (to_p)
	{
		if (prio)
			to_p->npriotasks++;
		else
			to_p->ntasks++;
	}
	else
		nflowing--;

	// fprintf(stderr,"pull from %s to %s\n", from_p?from_p->name:"none", to_p?to_p->name:"none");
	fxt_component_print_step(output, options, timestamp, workerid, 0, from_p, to_p);
}

void _starpu_fxt_component_finish(FILE *file)
{
	/* Javascript code from Anthony Simonet */
	fprintf(file, "\t\t<script>\n");
	fprintf(file, "\t\t\t$(function(){\n");
	fprintf(file, "\t\t\t\tsliderDiv = $('#slider') <!-- Alias -->\n");
	fprintf(file, "\t\t\t\tsliderDiv.slider({\n");
	fprintf(file, "\t\t\t\t\tvalue: 1,\n");
	fprintf(file, "\t\t\t\t\tmin: 1,\n");
	fprintf(file, "\t\t\t\t\tmax: %u,\n", global_state-1);
	fprintf(file, "\t\t\t\t\tstep: 1,\n");
	fprintf(file, "\t\t\t\t\tanimate: 'fast',\n");
	fprintf(file, "\t\t\t\t\tslide: function(event, ui){\n");
	fprintf(file, "\t\t\t\t\t\tvar l_value = sliderDiv.slider('option', 'value');\n");
	fprintf(file, "\t\t\t\t\t\t$('#etape').val(ui.value);\n");
	fprintf(file, "\t\t\t\t\t\tdocument.getElementById('et' + l_value).style.display = 'none';\n");
	fprintf(file, "\t\t\t\t\t\tdocument.getElementById('et' + ui.value).style.display = 'block';\n");
	fprintf(file, "\t\t\t\t\t}\n");
	fprintf(file, "\t\t\t\t});\n");
	fprintf(file, "\t\t\t\t$('#etape').val(sliderDiv.slider('value')); <!-- Initialisation au lancement de la page -->\n");
	fprintf(file, "\t\t\t\t$('#max').val(sliderDiv.slider('option', 'max'));\n");
	fprintf(file, "\t\t\t});\n");
	fprintf(file, "\t\t</script>\n");

	fprintf(file, "\t\t<div id ='slider'></div>\n");
	fprintf(file, "\t\t<center>\n");
	fprintf(file, "\t\t\t<p>\n");
	fprintf(file, "\t\t\t\t<input type='button' value='-100' onclick=\"changeState(-100);\"/>\n");
	fprintf(file, "\t\t\t\t<input type='button' value='<<' onmousedown=\"myVar = setInterval('changeState(-1)', 50)\" onmouseup=\"clearInterval(myVar)\" onmouseout=\"clearInterval(myVar)\"/>\n");
	fprintf(file, "\t\t\t\t<input type='button' value='<' onclick=\"changeState(-1);\"/>\n");
	fprintf(file, "\t\t\t\t<label for='etape'>State</label>\n");
	fprintf(file, "\t\t\t\t<input type='text' id='etape' size='3mm' readonly style='border:0;'>\n");
	fprintf(file, "\t\t\t\t<label for='max'>in</label>\n");
	fprintf(file, "\t\t\t\t<input type='text' id='max' size='3mm' readonly style='border:0;'>\n");
	fprintf(file, "\t\t\t\t<input type='button' value='>' onclick=\"changeState(1);\" />\n");
	fprintf(file, "\t\t\t\t<input type='button' value='>>' onmousedown=\"myVar = setInterval('changeState(1)', 50)\" onmouseup=\"clearInterval(myVar)\" onmouseout=\"clearInterval(myVar)\"/>\n");
	fprintf(file, "\t\t\t\t<input type='button' value='+100' onclick=\"changeState(100);\"/>\n");
	fprintf(file, "\t\t\t</p>\n");
	fprintf(file, "\t\t\t\t<span id='range'>Auto speed (state/s): 4</span>\n");
	fprintf(file, "\t\t\t\t<input type='range' id='autoRange' min='1' max='50' value='4' step='1' onchange=\"showValue(this.value); clearInterval(myVar);\"  />\n");
	fprintf(file, "\t\t\t\t<script>\n");
	fprintf(file, "\t\t\t\t\tdocument.getElementById('autoRange').value = 4;\n");
	fprintf(file, "\t\t\t\t\tfunction showValue(newValue)\n");
	fprintf(file, "\t\t\t\t\t{\n");
	fprintf(file, "\t\t\t\t\t\tdocument.getElementById('range').innerHTML='Auto speed (state/s): '+ newValue;\n");
	fprintf(file, "\t\t\t\t\t}\n");
	fprintf(file, "\t\t\t\t</script>\n");
	fprintf(file, "\t\t\t\t<input type='button' value='Auto' onclick=\"if(myVar){ clearInterval(myVar); myVar = null;}changeState(1); myVar = setInterval('changeState(1)', 1000/document.getElementById('autoRange').value);\"/>\n");
	fprintf(file, "\t\t\t\t<input type='button' value='Stop' onclick=\"clearInterval(myVar);\"/>\n");
	fprintf(file, "\t\t\t<p>\n");
	fprintf(file, "\t\t\t</p>\n");
	fprintf(file, "\t\t\t<FORM>\n");
	fprintf(file, "\t\t\t\t<span>Go to state</span>\n");
	fprintf(file, "\t\t\t\t<input type='text' name='setinput' id='input' value='' onKeyPress=\"if(event.keyCode == 13){ getInput(); javascript:this.value=''}\" onFocus=\"javascript:this.value=''\"/>\n");
	fprintf(file, "\t\t\t\t<input type='text' name='message' id='' value='' style='display:none'>\n"); /* Dummy input preventing the page from being refreshed when enter is pressed. */
	fprintf(file, "\t\t\t\t<input type='button' value='Go' onclick=\"getInput(); javascript:input.value=''\"/>\n");
	fprintf(file, "\t\t\t</FORM>\n");
	fprintf(file, "\t\t\t<br />\n");
	fprintf(file, "\t\t</center>\n");
	fprintf(file, "\t</body>\n");
	fprintf(file, "</html>\n");
}
#endif
