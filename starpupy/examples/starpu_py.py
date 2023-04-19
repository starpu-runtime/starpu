# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2023  Universite de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
from math import sqrt
import starpu
from starpu import starpupy
import time
import asyncio

def await_fut(fut):
    return fut.result()

try:
        starpu.init()
except Exception as e:
        print(e)
        exit(77)

############################################################################
#function no input no output print hello world
def hello():
	print ("Example 1:")
	print ("Hello, world!")

#############################################################################

#function no input no output
def func1():
	print ("Example 2:")
	print ("This is a function no input no output")

##############################################################################

#using decorator wrap the function no input no output
@starpu.delayed
def func1_deco():
	#time.sleep(1)
	print ("Example 3:")
	print ("This is a function no input no output wrapped by the decorator function")

##############################################################################

#function no input return a value
def func2():
	print ("Example 4:")
	return 12

###############################################################################

#function has 2 int inputs and 1 int output
def multi(a,b):
	print ("Example 5:")
	return a*b
#print(multi(2, 3))

###############################################################################

#function has 4 float inputs and 1 float output
def add(a,b,c,d):
	print ("Example 6:")
	return a+b+c+d
#print(add(1.2, 2.5, 3.6, 4.9))

###############################################################################

#function has 2 int inputs 1 float input and 1 float output 1 int output
def sub(a,b,c):
	print ("Example 7:")
	return a-b-c, a-b
#print(sub(6, 2, 5.9))

###############################################################################

#using decorator wrap the function with input
@starpu.delayed(name="test")
def add_deco(a,b,c):
	#time.sleep(1)
	print ("Example 8:")
	print ("This is a function with input and output wrapped by the decorator function:")
	return a+b+c

###############################################################################

#using decorator wrap the function with input
@starpu.delayed(color=1)
def sub_deco(x,a):
	print ("Example 9:")
	print ("This is a function with input and output wrapped by the decorator function:")
	return x-a

###############################################################################

async def main():
	#submit function "hello"
    fut = starpu.task_submit()(hello)
    await(fut)

    #submit function "func1"
    fut1 = starpu.task_submit()(func1)
    await(fut1)

    #apply starpu.delayed(func1_deco())
    await(func1_deco())

	#submit function "func2"
    fut2 = starpu.task_submit()(func2)
    res2 = await(fut2)
	#print the result of function
    print("This is a function no input and the return value is", res2)

    #submit function "multi"
    fut3 = starpu.task_submit()(multi, 2, 3)
    res3 = await(fut3)
    print("The result of function multi is :", res3)

	#submit function "add"
    fut4 = starpu.task_submit()(add, 1.2, 2.5, 3.6, 4.9)
    res4 = await(fut4)
    print("The result of function add is :", res4)

	#submit function "sub" but only provide function name
    fut5 = starpu.task_submit()(sub, 6, 2, 5.9)
    res5 = await(fut5)
    print("The result of function sub is:", res5)

	#apply starpu.delayed(add_deco)
    fut6 = add_deco(1,2,3)
    #res6 = await(fut6)
    #print("The result of function is", res6)

    #apply starpu.delayed(sub_deco)
    fut7 = sub_deco(fut6, 1)
    res7 = await(fut7)
    print("The first argument of this function is the result of Example 8")
    print("The result of function is", res7)

    fut8 = starpu.task_submit()("sqrt", 4)
    res8 = await(fut8)
    print("The result of function sqrt is:", res8)

try:
        asyncio.run(main())
except starpupy.error as e:
        print("No worker to execute the job")
        starpu.shutdown()
        exit(77)

starpu.shutdown()
#starpu.task_wait_for_all()
