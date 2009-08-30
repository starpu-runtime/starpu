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

#include "histo-svg.h"

static uint64_t absolute_start_time;
static uint64_t start_time;
static uint64_t absolute_end_time;
static uint64_t end_time;

static char *out_path;
static FILE *out_file;

static void add_region(worker_mode color, uint64_t start, uint64_t end, unsigned worker)
{
	float starty, endy, startx, endx;

	starty = SVG_BORDERY + (SVG_THICKNESS + SVG_GAP)*worker;
	endy = starty + SVG_THICKNESS;

	double ratio_start, ratio_end;
	
	ratio_start = (double)(start - start_time) / (double)(end_time - start_time);
	ratio_end = (double)(end - start_time) / (double)(end_time - start_time);

	startx = (float)(SVG_BORDERX + ratio_start*(SVG_WIDTH - 2*SVG_BORDERX)); 
	endx = (float)(SVG_BORDERX + ratio_end*(SVG_WIDTH - 2*SVG_BORDERX)); 

//	printf("startx %d endx %d  ratio %f %f starty %d endy %d\n", startx, endx, ratio_start, ratio_end, starty, endy);

	char *color_str;
	switch (color) {
		case WORKING:
			color_str = "green";
			break;
		case IDLE:
		default:
			color_str = "red";
			break;
	}

	fprintf(out_file, "<rect x=\"%fcm\" y=\"%fcm\" width=\"%fcm\" height=\"%fcm\" stroke-width=\"10\%%\" stroke=\"black\" fill=\"%s\"/>\n", startx, starty, endx - startx, endy -starty, color_str);

}

static void display_worker(event_list_t events, unsigned worker, char *worker_name)
{
	uint64_t prev = start_time;
	worker_mode prev_state = IDLE;

	fprintf(out_file, "<text x=\"%fcm\" y=\"%fcm\" font-size=\"%fcm\" fill=\"blue\" text-anchor=\"center\" > <tspan font-weight=\"bold\">%s</tspan></text>\n", 0.5f*SVG_BORDERX, SVG_BORDERY + (SVG_THICKNESS + SVG_GAP)*worker + 0.5f*SVG_THICKNESS, SVG_THICKNESS/4.0f, worker_name);

	event_itor_t i;
	for (i = event_list_begin(events);
		i != event_list_end(events);
		i = event_list_next(i))
	{
		add_region(prev_state, prev, i->time, worker);

		prev = i->time;
		prev_state = i->mode;
	}
}

static void display_workq_evolution(workq_list_t taskq, unsigned nworkers, unsigned maxq_size)
{
	float endy, starty;

	starty = SVG_BORDERY + (SVG_THICKNESS + SVG_GAP)*nworkers;
	endy = starty + SVG_THICKNESS;

	fprintf(out_file, "<line x1=\"%fcm\" y1=\"%fcm\" x2=\"%fcm\" y2=\"%fcm\" stroke=\"black\"  stroke-width=\"100\%%\" />\n", SVG_BORDERX, endy, SVG_WIDTH - SVG_BORDERX, endy);
	fprintf(out_file, "<line x1=\"%fcm\" y1=\"%fcm\" x2=\"%fcm\" y2=\"%fcm\" stroke=\"black\" stroke-width=\"100\%%\" />\n", SVG_BORDERX, starty, SVG_BORDERX, endy);

	float prevx, prevy;
	prevx = SVG_BORDERX;
	prevy = endy;

	workq_itor_t i;
	for (i = workq_list_begin(taskq);
		i != workq_list_end(taskq);
		i = workq_list_next(i))
	{
		float event_pos;
		double event_ratio;

		float y;

		event_ratio = ( i->time - start_time )/ (double)(end_time - start_time);
		event_pos = (SVG_BORDERX + event_ratio*(SVG_WIDTH - 2*SVG_BORDERX));

		double qratio;
		qratio = ((double)(i->current_size))/((double)maxq_size);

		y = ((double)endy - qratio *((double)SVG_THICKNESS));

		fprintf(out_file, "<line x1=\"%fcm\" y1=\"%fcm\" x2=\"%fcm\" y2=\"%fcm\" stroke=\"black\" stroke-width=\"30\%%\" />\n", prevx, prevy, event_pos, y);

		prevx = event_pos;
		prevy = y;
	}

	fprintf(out_file, "<line x1=\"%fcm\" y1=\"%fcm\" x2=\"%fcm\" y2=\"%fcm\" stroke=\"black\" stroke-width=\"100\%%\" />\n", prevx, prevy, SVG_BORDERX, endy);

}

void svg_output_file_init(void)
{
	/* create a new file */
	out_file = fopen(out_path, "w+");

	/* create some header for a valid svg file */
	fprintf(out_file, "<?xml version=\"1.0\" standalone=\"no\"?>\n");
	fprintf(out_file, "<!DOCTYPE svg PUBLIC \"-//W3C//DTD SVG 1.1//EN\" \"http://www.w3.org/Graphics/SVG/1.1/DTD/svg11.dtd\">\n");

	fprintf(out_file, "<svg width=\"%fcm\" height=\"%fcm\" version=\"1.1\" xmlns=\"http://www.w3.org/2000/svg\">\n", SVG_WIDTH, SVG_HEIGHT);

	/* a little description is not too much */
	fprintf(out_file, "<desc>Gantt diagram ...</desc>\n");
}

void svg_output_file_terminate(void)
{
	fprintf(out_file, "</svg>\n");

	/* close the file */	
	fclose(out_file);
}

void svg_engine_generate_output(event_list_t *events, workq_list_t taskq, char **worker_name,
			unsigned nworkers, unsigned maxq_size, 
			uint64_t _start_time, uint64_t _end_time, char *path)
{

	out_path = path;

	unsigned worker;

	start_time = _start_time;
	absolute_start_time = _start_time;
	end_time = _end_time;
	absolute_end_time = _end_time;

	svg_output_file_init();

	for (worker = 0; worker < nworkers; worker++)
	{
		display_worker(events[worker], worker, worker_name[worker]);
	}

	display_workq_evolution(taskq, nworkers, maxq_size);

	svg_output_file_terminate();
}
