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

/* example-start scrolledwin scrolledwin.c */

#include <stdio.h>
#include <gtk/gtk.h>
#include <stdlib.h>
#include "fxt-tool.h"
#include <assert.h>

#define GTK_WIDTH       800

#define GTK_THICKNESS   30
#define GTK_GAP         10

#define GTK_BORDERX             100
#define GTK_BORDERY             100


/* Backing pixmap for drawing area */
static GdkPixmap *pixmap = NULL;
static GdkGC *gc = NULL;
static GtkWidget *scrolled_window;
static GtkWidget *window;

static GdkColor blue;
static GdkColor red;
static GdkColor green;
static GdkColor white;
static GdkColor grey;
static GdkColor black;

static unsigned zoom = 1;

GtkWidget *drawing_area;


/* the actual trace data ... */
static event_list_t *events;
static workq_list_t taskq;
static char **worker_name;
static unsigned nworkers;
static unsigned maxq_size;
static uint64_t start_time;
static uint64_t end_time;

void trace_gantt(void);

static void init_colors(GtkWidget *area)
{
	GdkColormap *colormap;

	colormap = gtk_widget_get_colormap(area);
	if (!colormap)
		exit(1);

	green.red   = 0;
	green.green = 0xff * 0x100;
	green.blue  = 0;
	gdk_colormap_alloc_color(colormap, &green, FALSE, TRUE);

	blue.red   = 0;
	blue.green = 0;
	blue.blue  = 0xff * 0x100;
	gdk_colormap_alloc_color(colormap, &blue, FALSE, TRUE);

	red.red   = 0xff * 0x100;
	red.green = 0;
	red.blue  = 0;
	gdk_colormap_alloc_color(colormap, &red, FALSE, TRUE);

        white.red   = 0xff * 0x100;
        white.green = 0xff * 0x100;
        white.blue  = 0xff * 0x100;
        gdk_colormap_alloc_color(colormap, &white, FALSE, TRUE);

        grey.red   = 112 * 0x100;
        grey.green = 128 * 0x100;
        grey.blue  = 144 * 0x100;
        gdk_colormap_alloc_color(colormap, &grey, FALSE, TRUE);
	
        black.red   = 0;
        black.green = 0;
        black.blue  = 0;
        gdk_colormap_alloc_color(colormap, &white, FALSE, TRUE);
}



/* Create a new backing pixmap of the appropriate size */
static gint configure_event( GtkWidget         *widget,
                             GdkEventConfigure *event __attribute__ ((unused)))
{
  if (pixmap)
    gdk_pixmap_unref(pixmap);

  pixmap = gdk_pixmap_new(widget->window,
                          widget->allocation.width,
                          widget->allocation.height,
                          -1);
  gdk_draw_rectangle (pixmap,
                      widget->style->white_gc,
                      TRUE,
                      0, 0,
                      widget->allocation.width,
                      widget->allocation.height);

   trace_gantt();


	return TRUE;
}

static gint expose_event( GtkWidget      *widget,
                          GdkEventExpose *event )
{
	gdk_draw_pixmap(widget->window,
	                widget->style->fg_gc[GTK_WIDGET_STATE (widget)],
	                pixmap,
	                event->area.x, event->area.y,
	                event->area.x, event->area.y,
	                event->area.width, event->area.height);
	
	return FALSE;
}

/* Draw a rectangle on the screen */
static void draw_brush( GtkWidget *widget,
                        gdouble    x,
                        gdouble    y)
{
  GdkRectangle update_rect;

  update_rect.x = x - 5;
  update_rect.y = y - 5;
  update_rect.width = 10;
  update_rect.height = 10;
  gdk_draw_rectangle (pixmap,
                      widget->style->black_gc,
                      TRUE,
                      update_rect.x, update_rect.y,
                      update_rect.width, update_rect.height);
  gtk_widget_draw (widget, &update_rect);
}


static gint button_press_event( GtkWidget      *widget __attribute__ ((unused)),
                                GdkEventButton *event  __attribute__((unused)))
{
//  if (event->button == 1 && pixmap != NULL)
//    draw_brush (widget, event->x, event->y);
//
  return TRUE;
}

static gint motion_notify_event( GtkWidget *widget,
                                 GdkEventMotion *event )
{
  int x, y;
  GdkModifierType state;

  if (event->is_hint)
    gdk_window_get_pointer (event->window, &x, &y, &state);
  else
    {
      x = event->x;
      y = event->y;
      state = event->state;
    }

  if (state & GDK_BUTTON1_MASK && pixmap != NULL)
    draw_brush (widget, x, y);

  return TRUE;
}


void destroy( GtkWidget *widget __attribute__ ((unused)),
              gpointer   data __attribute__ ((unused)))
{
    gtk_main_quit();
}

static void gtk_add_region(worker_mode color, uint64_t start, uint64_t end, unsigned worker)
{
	unsigned long starty, endy, startx, endx;

	starty = GTK_BORDERY + (GTK_THICKNESS + GTK_GAP)*worker;
	endy = starty + GTK_THICKNESS;

	double ratio_start, ratio_end;
	ratio_start = (double)(start - start_time) / (double)(end_time - start_time);
	ratio_end = (double)(end - start_time) / (double)(end_time - start_time);

	startx = (unsigned long)(GTK_BORDERX + zoom*ratio_start*(GTK_WIDTH - 2*GTK_BORDERX));
	endx = (unsigned long)(GTK_BORDERX + zoom*ratio_end*(GTK_WIDTH - 2*GTK_BORDERX));

   GdkRectangle update_rect;

  update_rect.x = startx;
  update_rect.y = starty;
  update_rect.width = endx - startx;
  update_rect.height = endy - starty;

  gc = gdk_gc_new(drawing_area->window);
  assert(gc);



	switch (color) {
		case WORKING:
  			gdk_gc_set_foreground (gc, &green);
			break;
		case FETCHING:
  			gdk_gc_set_foreground (gc, &blue);
			break;
		case PUSHING:
  			gdk_gc_set_foreground (gc, &grey);
			break;
		case IDLE:
		default:
  			gdk_gc_set_foreground (gc, &red);
			break;
	}

  	gdk_draw_rectangle (pixmap,
  			      gc,
  	                    TRUE,
  	                    update_rect.x, update_rect.y,
  	                    update_rect.width, update_rect.height);
  	gtk_widget_draw (drawing_area, &update_rect);
}

static void gtk_display_worker(event_list_t worker_events, unsigned worker,
	char *worker_name __attribute__ ((unused)))
{
	uint64_t prev = start_time;
	worker_mode prev_state = IDLE;

	event_itor_t i;
	for (i = event_list_begin(worker_events);
		i != event_list_end(worker_events);
		i = event_list_next(i))
	{
		gtk_add_region(prev_state, prev, i->time, worker);

		prev = i->time;
		prev_state = i->mode;
	}
}

void trace_gantt(void)
{
//   GdkRectangle update_rect;
//
//  update_rect.x = 100 - 25;
//  update_rect.y = 100 - 25;
//  update_rect.width = zoom*GTK_WIDTH;
//  update_rect.height = zoom*GTK_WIDTH;
//
//  gc = gdk_gc_new(drawing_area->window);
//  assert(gc);
//  gdk_gc_set_foreground (gc, &green);
//
//  gdk_draw_rectangle (pixmap,
//  		      gc,
//                      TRUE,
//                      update_rect.x, update_rect.y,
//                      update_rect.width, update_rect.height);
//  gtk_widget_draw (drawing_area, &update_rect);

  unsigned worker;
  for (worker = 0; worker < nworkers; worker++)
  {
	 gtk_display_worker(events[worker], worker, worker_name[worker]);
  }
}

void refresh(void)
{

	unsigned drawing_area_height = 
		nworkers*GTK_THICKNESS + (nworkers-1)*GTK_GAP+2*GTK_BORDERY;
	unsigned drawing_area_width = 
		2*GTK_BORDERX + (GTK_WIDTH-2*GTK_BORDERX)*zoom;
		

	gtk_drawing_area_size (GTK_DRAWING_AREA (drawing_area),
		drawing_area_width , drawing_area_height);

	gdk_window_clear_area(drawing_area->window, 0, 0,
		drawing_area->allocation.width,
		drawing_area->allocation.height);

	gdk_draw_rectangle (pixmap,
		drawing_area->style->white_gc,
		TRUE, 0, 0,
		drawing_area->allocation.width,
		drawing_area->allocation.height);

	gdk_draw_pixmap(drawing_area->window,
		drawing_area->style->fg_gc[GTK_WIDGET_STATE (drawing_area)],
		pixmap,
		drawing_area->allocation.x, drawing_area->allocation.x,
		drawing_area->allocation.y, drawing_area->allocation.y,
		drawing_area->allocation.width,
		drawing_area->allocation.height);

	trace_gantt();	

}

void zoom_in_func( GtkWidget *widget  __attribute__ ((unused)),
			gpointer   data  __attribute__ ((unused)))
{
	GtkAdjustment *hadjustment;

	refresh();

	hadjustment = gtk_scrolled_window_get_hadjustment(GTK_SCROLLED_WINDOW (scrolled_window));
	float ratio = (hadjustment->value - GTK_BORDERX)/(GTK_BORDERX + zoom*(GTK_WIDTH - 2*GTK_BORDERX));

	zoom*=2;

	gtk_adjustment_set_value        (hadjustment, GTK_BORDERX + ratio*zoom*(GTK_WIDTH - 2*GTK_BORDERX));

	refresh();
}

void zoom_out_func( GtkWidget *widget __attribute__ ((unused)) , gpointer   data  __attribute__ ((unused)) )
{
	GtkAdjustment *hadjustment;

	hadjustment = gtk_scrolled_window_get_hadjustment(GTK_SCROLLED_WINDOW (scrolled_window));
	float ratio = (hadjustment->value - GTK_BORDERX)/(GTK_BORDERX + zoom*(GTK_WIDTH - 2*GTK_BORDERX));

	if (zoom > 1)
		zoom/=2;

	gtk_adjustment_set_value        (hadjustment, GTK_BORDERX + ratio*zoom*(GTK_WIDTH - 2*GTK_BORDERX));

	refresh();
}



int gtk_viewer_apps( int   argc, char *argv[], event_list_t *_events, 
                        workq_list_t _taskq, char **_worker_name, 
                        unsigned _nworkers, unsigned _maxq_size,
                        uint64_t _start_time, uint64_t _end_time)
{
	
	/* save the arguments */
	events = _events;
	taskq = _taskq;
	worker_name = _worker_name;
	nworkers = _nworkers;
	maxq_size = _maxq_size;
	start_time = _start_time;
	end_time = _end_time;
	
	GtkWidget *close_button, *zoom_in_button, *zoom_out_button;
	
	gtk_init (&argc, &argv);
	
	/* Create a new dialog window for the scrolled window to be
	 * packed into.  */
	window = gtk_dialog_new ();
	gtk_signal_connect (GTK_OBJECT (window), "destroy",
	    		(GtkSignalFunc) destroy, NULL);
	gtk_window_set_title (GTK_WINDOW (window), "GtkScrolledWindow example");
	gtk_container_set_border_width (GTK_CONTAINER (window), 0);
	gtk_widget_set_usize(window, STARPU_MIN(800, GTK_WIDTH),  STARPU_MIN(600, nworkers*GTK_THICKNESS + (nworkers-1)*GTK_GAP+2*GTK_BORDERY+100));
	
	/* create a new scrolled window. */
	scrolled_window = gtk_scrolled_window_new (NULL, NULL);
	
	gtk_container_set_border_width (GTK_CONTAINER (scrolled_window), 10);
	
	/* the policy is one of GTK_POLICY AUTOMATIC, or GTK_POLICY_ALWAYS.
	 * GTK_POLICY_AUTOMATIC will automatically decide whether you need
	 * scrollbars, whereas GTK_POLICY_ALWAYS will always leave the scrollbars
	 * there.  The first one is the horizontal scrollbar, the second, 
	 * the vertical. */
	gtk_scrolled_window_set_policy (GTK_SCROLLED_WINDOW (scrolled_window),
	                                GTK_POLICY_ALWAYS, GTK_POLICY_NEVER);
	/* The dialog window is created with a vbox packed into it. */								
	gtk_box_pack_start (GTK_BOX (GTK_DIALOG(window)->vbox), scrolled_window, 
	    		TRUE, TRUE, 0);
	gtk_widget_show (scrolled_window);
	
	/* the drawing box ... */
	drawing_area = gtk_drawing_area_new ();
	gtk_drawing_area_size (GTK_DRAWING_AREA (drawing_area), GTK_WIDTH , nworkers*GTK_THICKNESS + (nworkers-1)*GTK_GAP+2*GTK_BORDERY);
	
	
	gtk_scrolled_window_add_with_viewport (
	               GTK_SCROLLED_WINDOW (scrolled_window), drawing_area);
	
	
	init_colors(drawing_area);
	
	gtk_widget_show (drawing_area);
	
	/* Signals used to handle backing pixmap */
	gtk_signal_connect (GTK_OBJECT (drawing_area), "expose_event",
		(GtkSignalFunc) expose_event, NULL);
	gtk_signal_connect (GTK_OBJECT(drawing_area),"configure_event",
		(GtkSignalFunc) configure_event, NULL);
	
	/* Event signals */
	gtk_signal_connect (GTK_OBJECT (drawing_area), "motion_notify_event",
		(GtkSignalFunc) motion_notify_event, NULL);
	gtk_signal_connect (GTK_OBJECT (drawing_area), "button_press_event",
	    (GtkSignalFunc) button_press_event, NULL);
	
	gtk_widget_set_events (drawing_area, GDK_EXPOSURE_MASK
		| GDK_LEAVE_NOTIFY_MASK
	    | GDK_BUTTON_PRESS_MASK
	    | GDK_POINTER_MOTION_MASK
	    | GDK_POINTER_MOTION_HINT_MASK);
	
	
	/* Add a "close" button to the bottom of the dialog */
	close_button = gtk_button_new_with_label ("close");
	gtk_signal_connect_object (GTK_OBJECT (close_button), "clicked",
	    		       (GtkSignalFunc) gtk_widget_destroy,
	    		       GTK_OBJECT (window));
	
	/* this makes it so the button is the default. */
	GTK_WIDGET_SET_FLAGS (close_button, GTK_CAN_DEFAULT);
	gtk_box_pack_start (GTK_BOX (GTK_DIALOG (window)->action_area), close_button, TRUE, TRUE, 0);
	
	/* This grabs this button to be the default button. Simply hitting
	 * the "Enter" key will cause this button to activate. */
	gtk_widget_grab_default (close_button);
	gtk_widget_show (close_button);
	
	zoom_in_button = gtk_button_new_with_label ("zoom in");
	gtk_signal_connect_object (GTK_OBJECT (zoom_in_button), "clicked",
	    		       (GtkSignalFunc) zoom_in_func,
	    		       GTK_OBJECT (drawing_area));
	
	zoom_out_button = gtk_button_new_with_label ("zoom out");
	gtk_signal_connect_object (GTK_OBJECT (zoom_out_button), "clicked",
	    		       (GtkSignalFunc) zoom_out_func,
	    		       GTK_OBJECT (drawing_area));
	
	gtk_box_pack_start (GTK_BOX (GTK_DIALOG (window)->action_area), zoom_in_button, TRUE, TRUE, 0);
	gtk_box_pack_start (GTK_BOX (GTK_DIALOG (window)->action_area), zoom_out_button, TRUE, TRUE, 0);
	gtk_widget_show (zoom_in_button);
	gtk_widget_show (zoom_out_button);
	
	gtk_widget_show (window);
	
	gtk_main();
	
	return(0);
}
/* example-end */
