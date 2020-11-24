# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020       Université de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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
import starpu
import time
import asyncio
from math import sqrt
from math import log10
import numpy as np

#generate a list to store functions
g_func=[]

#function no input no output print hello world
def hello():
	print ("Example 1: Hello, world!")
g_func.append(starpu.joblib.delayed(hello)())

#function no input no output
def func1():
	print ("Example 2: This is a function no input no output")
g_func.append(starpu.joblib.delayed(func1)())

#function no input return a value
def func2():
	print ("Example 3:")
	return 12
g_func.append(starpu.joblib.delayed(func2)())
 
#function has 2 int inputs and 1 int output
def multi(a,b):
	res_multi=a*b
	print("Example 4: The result of ",a,"*",b,"is",res_multi)
	return res_multi
g_func.append(starpu.joblib.delayed(multi)(2, 3))

#function has 4 float inputs and 1 float output
def add(a,b,c,d):
	res_add=a+b+c+d
	print("Example 5: The result of ",a,"+",b,"+",c,"+",d,"is",res_add)
	return res_add
g_func.append(starpu.joblib.delayed(add)(1.2, 2.5, 3.6, 4.9))

#function has 2 int inputs 1 float input and 1 float output 1 int output
def sub(a,b,c):
	res_sub1=a-b-c
	res_sub2=a-b
	print ("Example 6: The result of ",a,"-",b,"-",c,"is",res_sub1,"and the result of",a,"-",b,"is",res_sub2)
	return res_sub1, res_sub2
g_func.append(starpu.joblib.delayed(sub)(6, 2, 5.9))

#the size of generator
N=100
# a=np.array([1,2,3,4,5,6,7,8,9,10])
# print(type(a))

def scal(a, t):
    return t*a

A=np.arange(N)


#starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="log")(starpu.joblib.delayed(log10)(i+1)for i in range(N))
# for x in [10, 100, 1000, 10000, 100000, 1000000]:
# 	for X2 in range(x, x*10, x):
# 		starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="log")(starpu.joblib.delayed(log10)(i+1)for i in range(X2))
# 		print(range(X2))

print("************************")
print("parallel Normal version:")
print("************************")
print("--input is iterable argument list, example 1")
starpu.joblib.Parallel(mode="normal", n_jobs=-2, perfmodel="first")(starpu.joblib.delayed(sqrt)(i**2)for i in range(N))

#################scikit test###################
DEFAULT_JOBLIB_BACKEND = starpu.joblib.get_active_backend()[0].__class__
class MyBackend(DEFAULT_JOBLIB_BACKEND):  # type: ignore
    def __init__(self, *args, **kwargs):
        self.count = 0
        super().__init__(*args, **kwargs)

    def start_call(self):
        self.count += 1
        return super().start_call()


starpu.joblib.register_parallel_backend('testing', MyBackend)

with starpu.joblib.parallel_backend("testing") as (ba, n_jobs):
	print("backend and n_jobs is", ba, n_jobs)
###############################################

print("--input is iterable argument list, example 2, with multi input")
a=np.arange(10)
b=np.arange(10)
starpu.joblib.Parallel(mode="normal", n_jobs=-2, perfmodel="first")(starpu.joblib.delayed(multi)(i,j) for i,j in zip(a,b))

print("--input is iterable function list")
starpu.joblib.Parallel(mode="normal", n_jobs=3, perfmodel="third")(g_func)

print("--input is numpy array")
starpu.joblib.Parallel(mode="normal", n_jobs=2, perfmodel="array")(starpu.joblib.delayed(scal)(2,A))

print("************************")
print("parallel Future version:")
print("************************")
async def main():
	print("--input is iterable argument list, example 1")
	fut1=starpu.joblib.Parallel(mode="future", n_jobs=-3, perfmodel="first")(starpu.joblib.delayed(sqrt)(i**2)for i in range(N))
	res1=await fut1
	#print(res1)

	print("--input is iterable argument list, example 2, with multi input")
	a=np.arange(10)
	b=np.arange(10)
	fut2=starpu.joblib.Parallel(mode="future", n_jobs=-3, perfmodel="second")(starpu.joblib.delayed(multi)(i,j) for i,j in zip(a,b))
	res2=await fut2
	#print(res2)

	print("--input is iterable function list")
	fut3=starpu.joblib.Parallel(mode="future", n_jobs=2)(g_func)
	res3=await fut3
	#print(res3)

	print("--input is numpy array")
	fut4=starpu.joblib.Parallel(mode="future", n_jobs=2, perfmodel="array")(starpu.joblib.delayed(scal)(2,A))
	res4=await fut4
	#print(res4)

asyncio.run(main())

starpu.perfmodel_plot(perfmodel="first",view=False)
starpu.perfmodel_plot(perfmodel="second",view=False)
starpu.perfmodel_plot(perfmodel="third",view=False)
