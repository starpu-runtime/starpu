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

#include "histo-flash.h"

static SWFMovie movie;

static uint64_t absolute_start_time;
static uint64_t start_time;
static uint64_t absolute_end_time;
static uint64_t end_time;

static SWFFont font;

void flash_engine_init(void)
{
	Ming_init();

	Ming_setScale(1.0);

	movie = newSWFMovie();

	SWFMovie_setBackground(movie, 0xff, 0xff, 0xff);
	SWFMovie_setDimension(movie, WIDTH, HEIGHT);

	const char *fontpath = "Sans.fdb";
	FILE *f = fopen(fontpath,"r");
	STARPU_ASSERT(f);

	font = loadSWFFontFromFile(f);
	if (font == NULL) {
		perror("could not open font :");
		exit(-1);
	}


}

#define PEN_WIDTH	0

static void add_region(worker_mode color, uint64_t start, uint64_t end, unsigned worker)
{
	unsigned starty, endy, startx, endx;

	starty = BORDERY + (THICKNESS + GAP)*worker;
	endy = starty + THICKNESS;

	double ratio_start, ratio_end;
	
	ratio_start = (double)(start - start_time) / (double)(end_time - start_time);
	ratio_end = (double)(end - start_time) / (double)(end_time - start_time);

	startx = (unsigned)(BORDERX + ratio_start*(WIDTH - 2*BORDERX)); 
	endx = (unsigned)(BORDERX + ratio_end*(WIDTH - 2*BORDERX)); 

//	printf("startx %d endx %d  ratio %f %f starty %d endy %d\n", startx, endx, ratio_start, ratio_end, starty, endy);

	int region_color[3];
		switch (color) {
			case WORKING:
				region_color[0] = 0;
				region_color[1] = 255;
				region_color[2] = 0;
				break;
			case IDLE:
			default:
				region_color[0] = 255;
				region_color[1] = 0;
				region_color[2] = 0;
				break;
		}


	SWFShape shape = newSWFShape();
//	SWFShape_setLine(shape, PEN_WIDTH, region_color[0], region_color[1], region_color[2], 255);
	SWFShape_setLine(shape, PEN_WIDTH, 0, 0, 0, 255);

	SWFFillStyle style= SWFShape_addSolidFillStyle(shape, region_color[0], region_color[1], region_color[2], 255);
	SWFShape_setRightFillStyle(shape, style);


	SWFShape_movePenTo(shape, startx, starty);
	SWFShape_drawLine(shape, endx-startx, 0);
	SWFShape_drawLine(shape, 0, endy-starty);
	SWFShape_drawLine(shape, (int)startx-(int)endx, 0);
	SWFShape_drawLine(shape, 0, -((int)endy-(int)starty));
	
	SWFMovie_add(movie, (SWFBlock)shape);
}

static void display_worker(event_list_t events, unsigned worker, char *worker_name)
{
	uint64_t prev = start_time;
	worker_mode prev_state = IDLE;

	SWFText namestr = newSWFText();
	SWFText_setFont(namestr, font);
	SWFText_setColor(namestr, 0, 0, 0, 0xff);
	SWFText_setHeight(namestr, 10);
	SWFText_moveTo(namestr, BORDERX/2 - strlen(worker_name), 
			BORDERY + (THICKNESS + GAP)*worker + THICKNESS/2);
	SWFText_addString(namestr, worker_name, NULL);

	SWFMovie_add(movie, (SWFBlock)namestr);

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

static char str_start[20];
static char str_end[20];

static void display_start_end_buttons(void)
{
	unsigned x_start, x_end, y;
	unsigned size = 15;

	sprintf(str_start, "start\n%lu", start_time-absolute_start_time);
	sprintf(str_end, "end\n%lu", end_time -absolute_start_time);

	x_start = BORDERX;
	x_end = WIDTH - BORDERX;
	y = BORDERY/2;

	SWFText text_start = newSWFText();
	SWFText_setFont(text_start, font);
	SWFText_setColor(text_start, 0, 0, 0, 0xff);
	SWFText_setHeight(text_start, size);
	SWFText_moveTo(text_start, x_start, y);
	SWFText_addString(text_start, str_start, NULL);

	SWFText text_end = newSWFText();
	SWFText_setFont(text_end, font);
	SWFText_setColor(text_end, 0, 0, 0, 0xff);
	SWFText_setHeight(text_end, size);
	SWFText_moveTo(text_end, x_end, y);
	SWFText_addString(text_end, str_end, NULL);

	SWFMovie_add(movie, (SWFBlock)text_start);
	SWFMovie_add(movie, (SWFBlock)text_end);

}

static void display_workq_evolution(workq_list_t taskq, unsigned nworkers, unsigned maxq_size)
{
	unsigned endy, starty;

	starty = BORDERY + (THICKNESS + GAP)*nworkers;
	endy = starty + THICKNESS;

	SWFShape shape = newSWFShape();
	SWFShape_setLine(shape, PEN_WIDTH, 0, 0, 0, 255);

//	SWFFillStyle style= SWFShape_addSolidFillStyle(shape, 0, 0, 0, 255);


	SWFShape_movePenTo(shape, BORDERX, endy);
	SWFShape_drawLine(shape, WIDTH - 2 *BORDERX, 0);
	SWFShape_movePenTo(shape, BORDERX, starty);
	SWFShape_drawLine(shape, 0, THICKNESS);
	
	SWFMovie_add(movie, (SWFBlock)shape);


	shape = newSWFShape();
	SWFShape_setLine(shape, 0, 0, 0, 0, 255);

	SWFShape_movePenTo(shape, BORDERX, endy);

	int prevx, prevy;
	prevx = BORDERX;
	prevy = endy;

	workq_itor_t i;
	for (i = workq_list_begin(taskq);
		i != workq_list_end(taskq);
		i = workq_list_next(i))
	{
		unsigned event_pos;
		double event_ratio;

		unsigned y;

		event_ratio = ( i->time - start_time )/ (double)(end_time - start_time);
		event_pos = (unsigned)(BORDERX + event_ratio*(WIDTH - 2*BORDERX));

		double qratio;
		qratio = ((double)(i->current_size))/((double)maxq_size);

		y = (unsigned)((double)endy - qratio *((double)THICKNESS));

		SWFShape_drawLine(shape, (int)event_pos - (int)prevx, (int)y - (int)prevy);
		prevx = event_pos;
		prevy = y;
	}

	SWFShape_drawLine(shape, (int)BORDERX - (int)prevx, (int)endy - (int)prevy);

	SWFMovie_add(movie, (SWFBlock)shape);

}

void flash_engine_generate_output(event_list_t *events, workq_list_t taskq, char **worker_name,
			unsigned nworkers, unsigned maxq_size, 
			uint64_t _start_time, uint64_t _end_time, char *path)
{
	unsigned worker;

	start_time = _start_time;
	absolute_start_time = _start_time;
	end_time = _end_time;
	absolute_end_time = _end_time;

	display_start_end_buttons();

	for (worker = 0; worker < nworkers; worker++)
	{
		display_worker(events[worker], worker, worker_name[worker]);
	}

	display_workq_evolution(taskq, nworkers, maxq_size);

	printf("save output ... \n");

	SWFMovie_save(movie, path);
}
