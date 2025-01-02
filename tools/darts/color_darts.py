# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2025   University of Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

"""
Return a color to each task when building a visualization
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import math
from colour import Color

def gradiant_color(gpu, order, number_task_gpu):
    red = 0
    green = 0
    blue = 0
    if gpu == 0:
        red = 1
    elif gpu == 1:
        green = 1
    elif gpu == 2:
        red = 73/255
        green = 116/255
        blue = 1
    elif gpu == 3:
        red = 1
        blue = 1
    elif gpu == 4:
        green = 1
        blue = 1
    elif gpu == 5:
        red = 1
        blue = 1
    elif gpu == 6:
        red = 1
        green = 0.5
        blue = 0.5
    elif gpu == 7:
        red = 0.5
        green = 1
        blue = 0.5
    else:
        red = 1/gpu
        green = 1/gpu
        blue = 1/gpu

    if red != 0:
        red = red - (red*order)/(number_task_gpu*1.3) # The multiplier help avoid darker colors
    if green != 0:
        green = green - (green*order)/(number_task_gpu*1.5)
    if blue != 0:
        blue = blue - (blue*order)/(number_task_gpu*1.5)

    if red != 0 and green == 0 and blue == 0:
        green = 0.3 - (red*order)/(number_task_gpu*1.2)
        blue = 0.3 - (red*order)/(number_task_gpu*1.2)
    elif red == 0 and green != 0 and blue == 0:
        red = 0.3 - (green*order)/(number_task_gpu*1.5)
        blue = 0.3 - (green*order)/(number_task_gpu*1.5)
    elif red == 0 and green == 0 and blue != 0:
        green = 0.3 - (blue*order)/(number_task_gpu*1.5)
        red = 0.3 - (blue*order)/(number_task_gpu*1.5)
    return(red, green, blue)

# Multiple colors for the same GPU
def gradiant_multiple_color(order, number_task_gpu, NGPU, current_gpu):
    pas = 500 # The pas is used to use a new color every pas tasks

    color_list = ((1,0,0), (0,1,0), (0,0,1), (1,1,0), (1,0,1), (0,1,1), (1,0.5,0.5), (0.5,1,0.5), (0.5,0.5,1), (1,0.25,0.25), (0.25,1,0.25), (0.25,0.25,1), (1,0.75,0.75), (0.75,0.75,1), (0.75,1,0.75), (1,0.5,0), (1,0,0.5), (0.5,1,0), (0.5,0,1), (0,1,0.5), (0,0.5,1), (1,0.25,0), (1,0,0.25), (0.25,1,0), (0.25,0,1), (0,1,0.25), (0,0.25,1))

    triplet_index = order//pas
    if triplet_index >= len(color_list):
        triplet_index = len(color_list) - 1

    r, g, blue = color_list[triplet_index]

    order = order%pas

    multiplier_to_lighten_up = 1.8
    if red != 0:
        red = red - (red*order)/(pas*multiplier_to_lighten_up)
    if green != 0:
        green = green - (green*order)/(pas*multiplier_to_lighten_up)
    if blue != 0:
        blue = blue - (blue*order)/(pas*multiplier_to_lighten_up)

    if red != 0 and green == 0 and blue == 0:
        green = 0.3 - (red*order)/(pas*1.2)
        blue = 0.3 - (red*order)/(pas*1.2)
    elif red == 0 and green != 0 and blue == 0:
        red = 0.3 - (green*order)/(pas*1.5)
        blue = 0.3 - (green*order)/(pas*1.5)
    elif red == 0 and green == 0 and blue != 0:
        green = 0.3 - (blue*order)/(pas*1.5)
        red = 0.3 - (blue*order)/(pas*1.5)

    return(red, green, blue)
