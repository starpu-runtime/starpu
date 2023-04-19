# StarPU --- Runtime system for heterogeneous multicore architectures.
#
# Copyright (C) 2020-2023  Universit'e de Bordeaux, CNRS (LaBRI UMR 5800), Inria
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

try:
    import numpy as np
except ModuleNotFoundError as e:
	print("Can't find \"Python3 NumPy\" module (consider running \"pip3 install numpy\" or refer to https://numpy.org/install/)")
	exit(77)

import starpu
import starpu.joblib
from starpu import starpupy
import time
import asyncio
from math import sqrt
from math import log10
import sys

try:
        starpu.init()
except Exception as e:
        print(e)
        exit(77)

def await_fut(fut):
    return fut.result()

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
def exp(a,b):
	res_exp=a**b
	print("Example 4: The result of ",a,"^",b,"is",res_exp)
	return res_exp
g_func.append(starpu.joblib.delayed(exp)(2, 3))

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

##########functions of array calculation###############
def scal(a, t):
	for i in range(len(t)):
		t[i]=t[i]*a
	return t

@starpu.access(t="RW")
def scal_np(a, t):
	for i in range(len(t)):
		t[i]=t[i]*a

@starpu.access(t1="RW")
def add_scal(a, t1, t2):
	for i in range(len(t1)):
		t1[i]=t1[i]*a+t2[i]
	#return t1

@starpu.access(t="RW")
def scal_arr(a, t):
	for i in range(len(t)):
		t[i]=t[i]*a[i]

def multi(a,b):
	res_multi=a*b
	return res_multi

def multi_2arr(a, b):
    for i in range(len(a)):
    	a[i]=a[i]*b[i]
    return a

@starpu.access(a="RW")
def multi_2np(a, b):
    for i in range(len(a)):
    	a[i]=a[i]*b[i]

def multi_list(l):
	res = []
	for (a,b) in l:
		res.append(a*b)
	return res

@starpu.access(t="RW")
def log10_arr(t):
	for i in range(len(t)):
		t[i]=log10(t[i])

########################################################

#################scikit test###################
# DEFAULT_JOBLIB_BACKEND = starpu.joblib.get_active_backend()[0].__class__
# class MyBackend(DEFAULT_JOBLIB_BACKEND):  # type: ignore
#         def __init__(self, *args, **kwargs):
#                 self.count = 0
#                 super().__init__(*args, **kwargs)

#         def start_call(self):
#                 self.count += 1
#                 return super().start_call()

# starpu.joblib.register_parallel_backend('testing', MyBackend)

# with starpu.joblib.parallel_backend("testing") as (ba, n_jobs):
# 	print("backend and n_jobs is", ba, n_jobs)
###############################################

N=100
# A=np.arange(N)
# B=np.arange(N)
# a=np.arange(N)
# b=np.arange(N, 2*N, 1)

displayPlot=False
listX=[10, 100]
for arg in sys.argv[1:]:
        if arg == "-long":
                listX = [10, 100, 1000, 10000, 100000, 1000000, 10000000]
        if arg == "-plot":
                displayPlot=True

for x in listX:
	for X in range(x, x*10, x):
		#print("X=",X)
		try :
			starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="log_list")(starpu.joblib.delayed(log10)(i+1)for i in range(X))
			A=np.arange(1,X+1,1)
			starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="log_arr")(starpu.joblib.delayed(log10_arr)(A))
		except starpupy.error as e:
			print("No worker to execute the job")
			exit(77)

print("************************")
print("parallel Normal version:")
print("************************")
print("--(sqrt)(i**2)for i in range(N)")
start_exec1=time.time()
start_cpu1=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="sqrt")(starpu.joblib.delayed(sqrt)(i**2)for i in range(N))
end_exec1=time.time()
end_cpu1=time.process_time()
print("the program execution time is", end_exec1-start_exec1)
print("the cpu execution time is", end_cpu1-start_cpu1)

print("--(multi)(i,j) for i,j in zip(a,b)")
a=np.arange(N)
b=np.arange(N, 2*N, 1)
start_exec2=time.time()
start_cpu2=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="multi")(starpu.joblib.delayed(multi)(i,j) for i,j in zip(a,b))
end_exec2=time.time()
end_cpu2=time.process_time()
print("the program execution time is", end_exec2-start_exec2)
print("the cpu execution time is", end_cpu2-start_cpu2)

print("--(scal_arr)((i for i in b), A)")
A=np.arange(N)
b=np.arange(N, 2*N, 1)
print("The input array is", A)
start_exec3=time.time()
start_cpu3=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="scal_arr")(starpu.joblib.delayed(scal_arr)((i for i in b), A))
end_exec3=time.time()
end_cpu3=time.process_time()
print("The return array is", A)
print("the program execution time is", end_exec3-start_exec3)
print("the cpu execution time is", end_cpu3-start_cpu3)

print("--(multi_list)((i,j) for i,j in zip(a,b))")
a=np.arange(N)
b=np.arange(N, 2*N, 1)
start_exec4=time.time()
start_cpu4=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="multi_list")(starpu.joblib.delayed(multi_list)((i,j) for i,j in zip(a,b)))
end_exec4=time.time()
end_cpu4=time.process_time()
print("the program execution time is", end_exec4-start_exec4)
print("the cpu execution time is", end_cpu4-start_cpu4)

print("--(multi_2arr)((i for i in a), (j for j in b))")
a=np.arange(N)
b=np.arange(N, 2*N, 1)
start_exec5=time.time()
start_cpu5=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="multi_2arr")(starpu.joblib.delayed(multi_2arr)((i for i in a), (j for j in b)))
end_exec5=time.time()
end_cpu5=time.process_time()
print("the program execution time is", end_exec5-start_exec5)
print("the cpu execution time is", end_cpu5-start_cpu5)

print("--(multi_2np)(A, B)")
# A=np.arange(N)
# B=np.arange(N, 2*N, 1)
n, m = 4, 5
A = np.arange(n*m).reshape(n, m)
B = np.arange(n*m, 2*n*m, 1).reshape(n, m)
print("The input arrays are A", A, "B", B)
start_exec6=time.time()
start_cpu6=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="multi_2arr")(starpu.joblib.delayed(multi_2np)(A, B))
end_exec6=time.time()
end_cpu6=time.process_time()
print("The return array is", A)
print("the program execution time is", end_exec6-start_exec6)
print("the cpu execution time is", end_cpu6-start_cpu6)

print("--(scal)(2, t=(j for j in a))")
a=np.arange(N)
start_exec7=time.time()
start_cpu7=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="scal")(starpu.joblib.delayed(scal)(2, t=(j for j in a)))
end_exec7=time.time()
end_cpu7=time.process_time()
print("the program execution time is", end_exec7-start_exec7)
print("the cpu execution time is", end_cpu7-start_cpu7)

print("--(scal_np)(2,A)")
A=np.arange(N)
print("The input is", A)
start_exec8=time.time()
start_cpu8=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="scal")(starpu.joblib.delayed(scal_np)(2,A))
end_exec8=time.time()
end_cpu8=time.process_time()
print("The return array is", A)
print("the program execution time is", end_exec8-start_exec8)
print("the cpu execution time is", end_cpu8-start_cpu8)

print("--(add_scal)(t1=A,t2=B,a=2)")
A=np.arange(N)
B=np.arange(N)
print("The input arrays are t1", A, "t2", B)
start_exec9=time.time()
start_cpu9=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="add_scal")(starpu.joblib.delayed(add_scal)(t1=A,t2=B,a=2))
end_exec9=time.time()
end_cpu9=time.process_time()
print("The return array is", A)
print("the program execution time is", end_exec9-start_exec9)
print("the cpu execution time is", end_cpu9-start_cpu9)


print("--input is iterable function list")
start_exec10=time.time()
start_cpu10=time.process_time()
starpu.joblib.Parallel(mode="normal", n_jobs=-1, perfmodel="func")(g_func)
end_exec10=time.time()
end_cpu10=time.process_time()
print("the program execution time is", end_exec10-start_exec10)
print("the cpu execution time is", end_cpu10-start_cpu10)

# def producer():
# 	for i in range(6):
# 		print('Produced %s' % i)
# 		yield i
#starpu.joblib.Parallel(n_jobs=2)(starpu.joblib.delayed(sqrt)(i) for i in producer())

print("************************")
print("parallel Future version:")
print("************************")
async def main():

	print("--(sqrt)(i**2)for i in range(N)")
	fut1=starpu.joblib.Parallel(mode="future", n_jobs=-1, perfmodel="sqrt")(starpu.joblib.delayed(sqrt)(i**2)for i in range(N))
	res1=await(fut1)
	print("The result is", sum(res1,[]))

	print("--(multi)(i,j) for i,j in zip(a,b)")
	a=np.arange(N)
	b=np.arange(N, 2*N, 1)
	print("The inputs are a", a, "b", b)
	fut2=starpu.joblib.Parallel(mode="future", n_jobs=-1, perfmodel="multi")(starpu.joblib.delayed(multi)(i,j) for i,j in zip(a,b))
	res2=await(fut2)
	print("The result is", sum(res2,[]))

	print("--(scal_arr)((i for i in b), A)")
	A=np.arange(N)
	b=np.arange(N, 2*N, 1)
	print("The input arrays are A", A, "b", b)
	fut3=starpu.joblib.Parallel(mode="future", n_jobs=-1, perfmodel="scal_arr")(starpu.joblib.delayed(scal_arr)((i for i in b), A))
	res3=await(fut3)
	#print("The return array is", np.concatenate(res3))
	print("The return array is", A)

	print("--(multi_list)((i,j) for i,j in zip(a,b))")
	a=np.arange(N)
	b=np.arange(N, 2*N, 1)
	print("The input lists are a", a, "b", b)
	fut4=starpu.joblib.Parallel(mode="future", n_jobs=-1, perfmodel="multi_list")(starpu.joblib.delayed(multi_list)((i,j) for i,j in zip(a,b)))
	res4=await(fut4)
	print("The result is", sum(res4,[]))

	print("--(multi_2arr)((i for i in a), (j for j in b))")
	a=np.arange(N)
	b=np.arange(N, 2*N, 1)
	print("The input lists are a", a, "b", b)
	fut5=starpu.joblib.Parallel(mode="future", n_jobs=-1, perfmodel="multi_2arr")(starpu.joblib.delayed(multi_2arr)((i for i in a), (j for j in b)))
	res5=await(fut5)
	print("The result is", sum(res5,[]))

	print("--(multi_2np)(b=B, a=A)")
	A=np.arange(N)
	B=np.arange(N, 2*N, 1)
	print("The input arrays are A", A, "B", B)
	fut6=starpu.joblib.Parallel(mode="future", n_jobs=-1, perfmodel="multi_2arr")(starpu.joblib.delayed(multi_2np)(b=B, a=A))
	res6=await(fut6)
	#print("The return array is", np.concatenate(res6))
	print("The return array is", A)


	print("--(scal)(2, (j for j in a))")
	a=np.arange(N)
	print("The input list is a", a)
	fut7=starpu.joblib.Parallel(mode="future", n_jobs=-1, perfmodel="scal")(starpu.joblib.delayed(scal)(2, (j for j in a)))
	res7=await(fut7)
	print("The result is", sum(res7,[]))

	print("--(scal_np)(2,t=A)")
	A=np.arange(N)
	print("The input array is", A)
	fut8=starpu.joblib.Parallel(mode="future", n_jobs=-1, perfmodel="scal")(starpu.joblib.delayed(scal_np)(2,t=A))
	res8=await(fut8)
	#print("The return array is", np.concatenate(res8))
	print("The return array is", A)

	print("--(scal)(2,A,B)")
	A=np.arange(N)
	B=np.arange(N)
	print("The input arrays are A", A, "B", B)
	fut9=starpu.joblib.Parallel(mode="future", n_jobs=-1, perfmodel="add_scal")(starpu.joblib.delayed(add_scal)(2,A,B))
	res9=await(fut9)
	#print("The return array is", np.concatenate(res9))
	print("The return array is", A)

	print("--input is iterable function list")
	fut10=starpu.joblib.Parallel(mode="future", n_jobs=-1)(g_func)
	res10=await(fut10)
	#print(res9)

try:
        asyncio.run(main())
except starpupy.error as e:
        starpu.shutdown()
        exit(77)

starpu.perfmodel_plot(perfmodel="sqrt",view=displayPlot)
starpu.perfmodel_plot(perfmodel="multi",view=displayPlot)
starpu.perfmodel_plot(perfmodel="scal_arr",view=displayPlot)
starpu.perfmodel_plot(perfmodel="multi_list",view=displayPlot)
starpu.perfmodel_plot(perfmodel="multi_2arr",view=displayPlot)
starpu.perfmodel_plot(perfmodel="scal",view=displayPlot)
starpu.perfmodel_plot(perfmodel="add_scal",view=displayPlot)
starpu.perfmodel_plot(perfmodel="func",view=displayPlot)

starpu.perfmodel_plot(perfmodel="log_list",view=displayPlot)
starpu.perfmodel_plot(perfmodel="log_arr",view=displayPlot)

starpu.shutdown()
