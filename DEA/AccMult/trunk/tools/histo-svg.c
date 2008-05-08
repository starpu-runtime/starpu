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

	fprintf(out_file, "<rect x=\"%fcm\" y=\"%fcm\" width=\"%fcm\" height=\"%fcm\" stroke-width=\"10\%\" stroke=\"black\" fill=\"%s\"/>\n", startx, starty, endx - startx, endy -starty, color_str);

//
//	SWFShape shape = newSWFShape();
//	SWFShape_setLine(shape, PEN_SVG_WIDTH, 0, 0, 0, 255);
//
//	SWFFillStyle style= SWFShape_addSolidFillStyle(shape, region_color[0], region_color[1], region_color[2], 255);
//	SWFShape_setRightFillStyle(shape, style);
//
//
//	SWFShape_movePenTo(shape, startx, starty);
//	SWFShape_drawLine(shape, endx-startx, 0);
//	SWFShape_drawLine(shape, 0, endy-starty);
//	SWFShape_drawLine(shape, (int)startx-(int)endx, 0);
//	SWFShape_drawLine(shape, 0, -((int)endy-(int)starty));
//	
//	SWFMovie_add(movie, (SWFBlock)shape);

	

}

static void display_worker(event_list_t events, unsigned worker, char *worker_name)
{
	uint64_t prev = start_time;
	worker_mode prev_state = IDLE;

//	SWFText namestr = newSWFText();
//	SWFText_setFont(namestr, font);
//	SWFText_setColor(namestr, 0, 0, 0, 0xff);
//	SWFText_setHeight(namestr, 10);
//	SWFText_moveTo(namestr, BORDERX/2 - strlen(worker_name), 
//			SVG_BORDERY + (SVG_THICKNESS + SVG_GAP)*worker + SVG_THICKNESS/2);
//	SWFText_addString(namestr, worker_name, NULL);
//
//	SWFMovie_add(movie, (SWFBlock)namestr);

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

//char str_start[20];
//char str_end[20];
//
//static void display_start_end_buttons(void)
//{
//	unsigned x_start, x_end, y;
//	unsigned size = 15;
//
//	sprintf(str_start, "start\n%lu", start_time-absolute_start_time);
//	sprintf(str_end, "end\n%lu", end_time -absolute_start_time);
//
//	x_start = SVG_BORDERX;
//	x_end = SVG_WIDTH - SVG_BORDERX;
//	y = SVG_BORDERY/2;
//
//	SWFText text_start = newSWFText();
//	SWFText_setFont(text_start, font);
//	SWFText_setColor(text_start, 0, 0, 0, 0xff);
//	SWFText_setHeight(text_start, size);
//	SWFText_moveTo(text_start, x_start, y);
//	SWFText_addString(text_start, str_start, NULL);
//
//	SWFText text_end = newSWFText();
//	SWFText_setFont(text_end, font);
//	SWFText_setColor(text_end, 0, 0, 0, 0xff);
//	SWFText_setHeight(text_end, size);
//	SWFText_moveTo(text_end, x_end, y);
//	SWFText_addString(text_end, str_end, NULL);
//
//	SWFMovie_add(movie, (SWFBlock)text_start);
//	SWFMovie_add(movie, (SWFBlock)text_end);
//
//}

static void display_workq_evolution(workq_list_t taskq, unsigned nworkers, unsigned maxq_size)
{
	unsigned endy, starty;

	starty = SVG_BORDERY + (SVG_THICKNESS + SVG_GAP)*nworkers;
	endy = starty + SVG_THICKNESS;

//	SWFShape shape = newSWFShape();
//	SWFShape_setLine(shape, PEN_SVG_WIDTH, 0, 0, 0, 255);
//
//
//	SWFShape_movePenTo(shape, SVG_BORDERX, endy);
//	SWFShape_drawLine(shape, SVG_WIDTH - 2 *SVG_BORDERX, 0);
//	SWFShape_movePenTo(shape, SVG_BORDERX, starty);
//	SWFShape_drawLine(shape, 0, SVG_THICKNESS);
//	
//	SWFMovie_add(movie, (SWFBlock)shape);
//
//
//	shape = newSWFShape();
//	SWFShape_setLine(shape, 0, 0, 0, 0, 255);
//
//	SWFShape_movePenTo(shape, SVG_BORDERX, endy);
//
	int prevx, prevy;
	prevx = SVG_BORDERX;
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
		event_pos = (unsigned)(SVG_BORDERX + event_ratio*(SVG_WIDTH - 2*SVG_BORDERX));

		double qratio;
		qratio = ((double)(i->current_size))/((double)maxq_size);

		y = (unsigned)((double)endy - qratio *((double)SVG_THICKNESS));

		//SWFShape_drawLine(shape, (int)event_pos - (int)prevx, (int)y - (int)prevy);
		prevx = event_pos;
		prevy = y;
	}

//	SWFShape_drawLine(shape, (int)SVG_BORDERX - (int)prevx, (int)endy - (int)prevy);
//
//	SWFMovie_add(movie, (SWFBlock)shape);

}

// 
// <svg width="10cm" height="3cm" viewBox="0 0 1000 300"
//      xmlns="http://www.w3.org/2000/svg" version="1.1">
//   <desc>Example tspan01 - using tspan to change visual attributes</desc>
//   <g font-family="Verdana" font-size="45" >
//     <text x="200" y="150" fill="blue" >
//       You are
//         <tspan font-weight="bold" fill="red" >not</tspan>
//       a banana.
//     </text>
//   </g>
//   <!-- Show outline of canvas using 'rect' element -->
//   <!--
//   <rect x="1" y="1" width="998" height="298"
//         fill="none" stroke="blue" stroke-width="2" /> -->
// </svg>
// 

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
