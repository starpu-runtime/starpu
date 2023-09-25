# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2023  Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
#
# StarPU is free software; you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation; either version 2.1 of the License, or (at
# your option) any later version.
#
# StarPU is distributed in the hope that it will be useful, but
# WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
#
# See the GNU Lesser General Public License in COPYING.LGPL for more details.
#

# Return a color to each task when building a visualization

import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from colour import Color

def gradiant_color(gpu, order, number_task_gpu):
	r = 0
	g = 0
	b = 0
	if (gpu == 0):
		r = 1
	elif (gpu == 1):
		g = 1
	elif (gpu == 2):
		r = 73/255
		g = 116/255
		b = 1
	elif (gpu == 3):
		r = 1
		b = 1
	elif (gpu == 4):
		g = 1
		b = 1
	elif (gpu == 5):
		r = 1
		b = 1
	elif (gpu == 6):
		r = 1
		g = 0.5
		b = 0.5
	elif (gpu == 7):
		r = 0.5
		g = 1
		b = 0.5
	else:
		r = 1/gpu
		g = 1/gpu
		b = 1/gpu
		
	if (r != 0): 
		r = r - (r*order)/(number_task_gpu*1.3) # The multiplier help avoid darker colors
	if (g != 0):
		g = g - (g*order)/(number_task_gpu*1.5)
	if (b != 0):
		b = b - (b*order)/(number_task_gpu*1.5)

	if (r != 0 and g == 0 and b == 0):
		g = 0.3 - (r*order)/(number_task_gpu*1.2)
		b = 0.3 - (r*order)/(number_task_gpu*1.2)
	elif (r == 0 and g != 0 and b == 0):
		r = 0.3 - (g*order)/(number_task_gpu*1.5)
		b = 0.3 - (g*order)/(number_task_gpu*1.5)
	elif (r == 0 and g == 0 and b != 0):
		g = 0.3 - (b*order)/(number_task_gpu*1.5)
		r = 0.3 - (b*order)/(number_task_gpu*1.5)
	return(r, g, b)
  
# Multiple colors for the same GPU
def gradiant_multiple_color(order, number_task_gpu, NGPU, current_gpu):
	pas = 500 # The pas is used to use a new color every pas tasks
	
	color_list = ((1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,0.5,0.5), (0.5,1,0.5), (0.5,0.5,1), (1,0.25,0.25), (0.25,1,0.25), (0.25,0.25,1), (1,0.75,0.75), (0.75,0.75,1), (0.75,1,0.75), (1,0.5,0), (1,0,0.5), (0.5,1,0), (0.5,0,1), (0,1,0.5), (0,0.5,1), (1,0.25,0), (1,0,0.25), (0.25,1,0), (0.25,0,1), (0,1,0.25), (0,0.25,1))

	triplet_index = order//pas
	if triplet_index >= len(color_list):
		triplet_index = len(color_list) - 1
			
	r, g, b = color_list[triplet_index]
		
	order = order%pas
		
	multiplier_to_lighten_up = 1.8
	if (r != 0): 
		r = r - (r*order)/(pas*multiplier_to_lighten_up)
	if (g != 0):
		g = g - (g*order)/(pas*multiplier_to_lighten_up)
	if (b != 0):
		b = b - (b*order)/(pas*multiplier_to_lighten_up)

	if (r != 0 and g == 0 and b == 0):
		g = 0.3 - (r*order)/(pas*1.2)
		b = 0.3 - (r*order)/(pas*1.2)
	elif (r == 0 and g != 0 and b == 0):
		r = 0.3 - (g*order)/(pas*1.5)
		b = 0.3 - (g*order)/(pas*1.5)
	elif (r == 0 and g == 0 and b != 0):
		g = 0.3 - (b*order)/(pas*1.5)
		r = 0.3 - (b*order)/(pas*1.5)

	return(r, g, b)
