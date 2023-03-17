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
from starpu import starpupy
from starpu import Handle
from starpu import HandleNumpy
import asyncio
import time
import array
import struct

try:
        starpu.init()
except Exception as e:
        print(e)
        exit(77)

if starpupy.worker_get_count_by_type(starpu.STARPU_MPI_MS_WORKER) >= 1 or starpupy.worker_get_count_by_type(starpu.STARPU_TCPIP_MS_WORKER) >= 1:
	print("This program does not work in MS mode")
	starpu.shutdown()
	exit(77)

def show(x, y):
    print("Function printing:", x, y)

def add(x, y):
	print ("Example add(x, y):")
	return x + y

# create Handle objects
x=2
y=3
x_h = Handle(x)
y_h = Handle(y)

print("*************************")
print("constant handle:")
print("*************************")
# show function returns Handle
ret_h1 = starpu.task_submit(ret_handle=True)(show, "first argument is:", x_h)
print("show funtion returns:", ret_h1.get())

# return value is Handle
res1 = starpu.task_submit(ret_handle=True)(add, x_h, y_h)
print("result of Handle(2)+Handle(3) is:", res1.get())

# return value is Handle
res2 = starpu.task_submit(ret_handle=True)(add, res1, y_h)
print("result of res1+Handle(3) is:", res2.get())

# show function returns Handle
ret_h2 = starpu.task_submit(ret_handle=True)(show, res1, res2)

print("*************************************")
print("constant handle return in parameter:")
print("*************************************")

ret = Handle(0)
print("before calling function, ret value is:", ret.get())
# return value as parameter
ret_n = starpu.task_submit(ret_param=True)(add, ret, x_h, y_h)
print("result of Handle(2)+Handle(3) is:", ret.get())
print("return value of task_submit is:", ret_n)
assert ret.get() == x+y

x_h.unregister()
y_h.unregister()

ret_h1.unregister()
ret_h2.unregister()
res2.unregister()
ret.unregister()

##############################################################################################
print("*************************")
print("Numpy array handle:")
print("*************************")
def scal(x, t):
	for i in range(len(t)):
		t[i] = t[i] * x
	print ("Example scal(scalar, array):")

t = np.arange(10)

# create Handle object for Numpy array
t_h = Handle(t)

# return value is Handle
res3 = starpu.task_submit(ret_handle=True)(scal, 2, t_h)
print("result of scal(2, Handle(np.arange(10)) is:", t_h.get())

# show function returns Future
async def main():
	res_fut1 = starpu.task_submit()(show, res1, t_h)
	await res_fut1
asyncio.run(main())

t_h.unregister()
res1.unregister()
res3.unregister()

######################
def arr_add(a,b):
	for i in range(np.size(a)):
		a[i] = a[i] + b[i]

a = np.array([1, 2, 3])
b = np.array([4, 5, 6])

# create Handle objects
a_h = Handle(a)
b_h = Handle(b)

# two array element addition
res4 = starpu.task_submit(ret_handle=True)(arr_add, a_h, b_h)
print("result of adding two Handle(numpy.array) is:", a_h.get())

a_h.unregister()
b_h.unregister()

res4.unregister()

#######################
def multi(x, y):
	print ("Example multi(x, y):")
	np.multiply(x, y, out=x)

c = np.array([[1, 2], [3, 4]])
d = np.array([[2, 2], [2, 2]])

# create Handle objects
c_h = Handle(c)
d_h = Handle(d)

# two array element multiplication
res5 = starpu.task_submit(ret_handle=True)(multi, c_h, d_h)
print("result of multiplying two Handle(numpy.array) is:", c_h.get())

########################
@starpu.access(x="RW")
def matrix_multi(x, y):
	print ("Example matrix_multi(x, y):")
	np.dot(x, y, out=x)

# two array matrix multiplication
res6 = starpu.task_submit(ret_handle=True)(matrix_multi, c_h, d_h)
print("result of matrix multiplying two Handle(numpy.array) is:", c_h.get())

# two array matrix multiplication (inverse order)
res7 = starpu.task_submit(ret_handle=True)(matrix_multi, d_h, c_h)
print("result of matrix multiplying two Handle(numpy.array) is:", d_h.get())

c_h.unregister()
d_h.unregister()

res5.unregister()
res6.unregister()
res7.unregister()

###################################empty Numpy array handle#####################################
print("*************************")
print("empty Numpy array handle:")
print("*************************")
a1 = np.array([1, 2, 3, 4])
a2 = np.array([[1, 2, 3], [4, 5, 6]])
a3 = np.array([[[1, 2, 3], [4, 5, 6]],[[7, 8, 9], [10, 11, 12]]])

# create Handle objects
a1_h = Handle(a1)
a2_h = Handle(a2)
a3_h = Handle(a3)

a1_r = a1_h.acquire(mode='R')
print("original 1-dimension array is:", a1_r)
a1_h.release()
a2_r = a2_h.acquire(mode='R')
print("original 2-dimension array is:", a2_r)
a2_h.release()
a3_r = a3_h.acquire(mode='R')
print("original 3-dimension array is:", a3_r)
a3_h.release()

@starpu.access(b="W")
def assign(a,b):
	for i in range(min(np.size(a), np.size(b))):
		b[i] = a[i]

@starpu.access(b="W")
def assign2(a,b):
	for i in range(min(np.size(a,0), np.size(b,0))):
		for j in range(min(np.size(a,1), np.size(b,1))):
			b[i][j] = a[i][j]

@starpu.access(b="W")
def assign3(a,b):
	for i in range(min(np.size(a,0), np.size(b,0))):
		for j in range(min(np.size(a,1), np.size(b,1))):
			for k in range(min(np.size(a,2), np.size(b,2))):
				b[i][j][k] = a[i][j][k]

# generate empty arrays Handle object using HandleNumpy
# 1-dimension
e1_h = HandleNumpy(a1.shape, a1.dtype)

res8 = starpu.task_submit(ret_handle=True)(assign, a1_h, e1_h)
e1_r = e1_h.acquire(mode='RW')
print("assigned 1-dimension array is:", e1_r)
# e1_h is writeable, we modify the first element
e1_r[0] = 100
print("the first element of 1-dimension array is modified to 100:", e1_r)
e1_h.release()

# 2-dimension
e2_h = HandleNumpy(a2.shape, a2.dtype)
res9 = starpu.task_submit(ret_handle=True)(assign2, a2_h, e2_h)
e2_r = e2_h.acquire(mode='R')
print("assigned 2-dimension array is", e2_r)
e2_h.release()

# 3-dimension
e3_h = HandleNumpy(a3.shape, a3.dtype)
res10 = starpu.task_submit(ret_handle=True)(assign3, a3_h, e3_h)
e3_r = e3_h.acquire(mode='R')
print("assigned 3-dimension array is", e3_r)
e3_h.release()

a1_h.unregister()
a2_h.unregister()
a3_h.unregister()
e1_h.unregister()
e2_h.unregister()
e3_h.unregister()

res8.unregister()
res9.unregister()
res10.unregister()

##################################bytes handle############################################
print("*************************")
print("bytes handle:")
print("*************************")
bt1 = bytes([1,2])
bt2 = bytes([3,4])

bt1_h = Handle(bt1)
bt2_h = Handle(bt2)

bt1_r = bt1_h.acquire(mode='R')
print("first bytes object is", bt1_r)
bt1_h.release()

bt2_r = bt2_h.acquire(mode='R')
print("second bytes object is", bt2_r)
bt2_h.release()

ret_bt1 = starpu.task_submit(ret_handle=True)(add, bt1_h, bt2_h)
print("result of appending two bytes: ", ret_bt1.get())

def bytes_add(x, y):
	z = bytearray(len(x))
	for i in range (len(x)):
		z[i] = x[i] + y[i]
	return bytes(z)

ret_bt2 = starpu.task_submit(ret_handle=True)(bytes_add, bt1_h, bt2_h)
print("result of adding two bytes elements: ", ret_bt2.get())

bt1_h.unregister()
bt2_h.unregister()

ret_bt1.unregister()
ret_bt2.unregister()

####################################bytearray handle#########################################
print("*************************")
print("bytearray handle:")
print("*************************")
bta1 = bytearray([1,2])
bta2 = bytearray([3,4])

bta1_h = Handle(bta1)
bta2_h = Handle(bta2)

bta1_r = bta1_h.acquire(mode='RW')
print("first bytearray object is", bta1_r)
bta1[0] = 0
bta1_h.release()
bta11_r = bta1_h.acquire(mode='R')
print("first bytearray object is modified", bta11_r)
bta1_h.release()

bta2_r = bta2_h.acquire(mode='R')
print("second bytearray object is", bta2_r)
bta2_h.release()

def bytearray_add(x, y):
	z = bytearray(len(x))
	for i in range (len(x)):
		z[i] = x[i] + y[i]
	return z

ret_bta1 = starpu.task_submit(ret_handle=True)(bytearray_add, bta1_h, bta2_h)
print("result of adding two bytearray elements: ", ret_bta1.get())

bta1_h.unregister()
bta2_h.unregister()

ret_bta1.unregister()

##################################array.array handle##########################################
print("*************************")
print("array.array handle:")
print("*************************")
arr1 = array.array('i', [1, 2, 3, 4])
arr2 = array.array('i', [2, 2, 2, 2])
arr3 = array.array('f', [4.5, 5.5, 6.5])

arr4 = array.array('u', 'hello')

def arrarr_add(x, y):
	for i in range (len(x)):
		x[i] = x[i] + y[i]
	#time.sleep(1)
	return x

def arrarr_multi(x, y):
	for i in range (len(x)):
		x[i] = x[i] * y[i]
	return x

def arrarr_scal(x, s):
	for i in range (len(x)):
		x[i] = x[i] * s
	return x

arr1_h = Handle(arr1)
arr1_r = arr1_h.acquire(mode='RW')
print("first array.array object is", arr1_r)
arr1[0] = 0
arr1_h.release()
arr11_r = arr1_h.acquire(mode='R')
print("first array.array object is modified", arr11_r)
arr1_h.release()

arr2_h = Handle(arr2)
arr2_r = arr2_h.acquire(mode='R')
print("second array.array object is", arr2_r)
arr2_h.release()

arr3_h = Handle(arr3)
arr3_r = arr3_h.acquire(mode='R')
print("third array.array object is", arr3_r)
arr3_h.release()

arr4_h = Handle(arr4)
arr4_r = arr4_h.acquire(mode='R')
print("fourth array.array object is", arr4_r)
arr4_h.release()

ret_arr1 = starpu.task_submit(ret_handle=True)(arrarr_add, arr1_h, arr2_h)
print("result of adding two array.array elements: ", ret_arr1.get())

ret_arr2 = starpu.task_submit(ret_handle=True)(arrarr_multi, arr1_h, arr2_h)
print("result of multiplying two array.array elements: ", ret_arr2.get())

ret_arr3 = starpu.task_submit(ret_handle=True)(arrarr_scal, arr3_h, 2)
print("result of multiplying array.array element by a scalar: ", ret_arr3.get())

arr1_h.unregister()
arr2_h.unregister()
arr3_h.unregister()
arr4_h.unregister()

ret_arr1.unregister()
ret_arr2.unregister()
ret_arr3.unregister()

##################################memoryview handle###########################################
print("*************************")
print("memoryview handle:")
print("*************************")
m1 = memoryview(bytearray("hello", 'utf-8'))
m1_tb = m1.tobytes()
print("m1 to bytes is", m1_tb)

m2 = memoryview(array.array('i', [1, 2, 3, 4]))
m2_tl = m2.tolist()
print("m2 to list is", m2_tl)

m3 = memoryview(array.array('u', 'hello'))

m1_h = Handle(m1)
print("m1 is", m1_h.acquire(mode='RW'))
m1[0] = 100
m1_h.release()

print("m1 to bytes after modifying is", m1_tb)
print("m1 after modifying is", m1_h.acquire(mode='RW'))
m1_h.release()

m2_h = Handle(m2)
print("m2 is", m2_h.acquire(mode='R'))
m2_h.release()

m3_h = Handle(m3)
print("m3 is", m3_h.acquire(mode='R'))
m3_h.release()

# multi dimension
def mem_show(x):
	print("memory is", x)

buf = struct.pack("L"*12, *list(range(12)))
x = memoryview(buf)
# 2-dimension
y = x.cast('L', shape=[3,4])
# 3-dimension
z = x.cast('L', shape=[2,3,2])
print(y.tolist())
print(z.tolist())

y_h = Handle(y)
ret_m1 = starpu.task_submit(ret_handle=True)(mem_show, y_h)

print("y is", y_h.acquire(mode='R'))
y_h.release()

z_h = Handle(z)
ret_m2 = starpu.task_submit(ret_handle=True)(mem_show, z_h)

print("z is", z_h.acquire(mode='R'))
z_h.release()

m1_h.unregister()
m2_h.unregister()
m3_h.unregister()

y_h.unregister()
z_h.unregister()

ret_m1.unregister()
ret_m2.unregister()

#####################################access mode annotation###################################
print("*************************")
print("access mode annotation:")
print("*************************")
a = np.array([1, 2, 3, 4])
a_h = Handle(a)
e_h = HandleNumpy(a.shape, a.dtype)

a_r = a_h.acquire(mode='R')
print("original array is:", a_r)
a_h.release()

######access#####
print("------------------")
print("access decorator:")
print("------------------")
@starpu.access(a="R", b="W")
def assign(a,b):
	for i in range(min(np.size(a), np.size(b))):
		b[i]=a[i]

res11 = starpu.task_submit(ret_handle=True)(assign, a_h, e_h)

e_r = e_h.acquire(mode='RW')
print("assigned 1-dimension array is:", e_r)
e_h.release()

######delayed#######
print("------------------")
print("delayed decorator:")
print("------------------")
@starpu.delayed(ret_handle=True, a="R", b="W")
def assign(a,b):
	for i in range(min(np.size(a), np.size(b))):
		b[i]=a[i]

res12 = assign(a_h, e_h)

e_r = e_h.acquire(mode='RW')
print("assigned 1-dimension array is:", e_r)
e_h.release()

######set access######
print("------------------")
print("access function:")
print("------------------")
def assign(a,b):
	for i in range(min(np.size(a), np.size(b))):
		b[i]=a[i]

assign_access=starpu.set_access(assign, a="R", b="W")
res13 = starpu.task_submit(ret_handle=True)(assign_access, a_h, e_h)

e_r = e_h.acquire(mode='RW')
print("assigned 1-dimension array is:", e_r)
e_h.release()

a_h.unregister()
e_h.unregister()

res11.unregister()
res12.unregister()
res13.unregister()

#######################Numpy without explicit handle############################
print("*******************************")
print("Numpy without explicit handle:")
print("*******************************")
arrh1 = np.array([1, 2, 3])
arrh2 = np.array([4, 5, 6])

@starpu.access(a="RW", b="R")
def np_add(a, b):
	#time.sleep(2)
	for i in range(np.size(a)):
		a[i] = a[i] + b[i]

print("First argument before task submitting is", starpu.acquire(arrh1, mode='R'))
#a[0]=100
starpu.release(arrh1)
# without explicit handle
res14 = starpu.task_submit(ret_handle=True)(np_add, arrh1, arrh2)

print("First argument after task submitting is", starpu.acquire(arrh1, mode='R'))
starpu.release(arrh1)

# it's mandatory to call unregister when the argument is no longer needed to access, but it's not obligatory, calling starpupy.shutdown() in the end is enough, which will unregister all no-explicit handle
starpu.unregister(arrh1)

res14.unregister()

#######################Numpy without using handle###############################
print("*******************************")
print("Numpy without using handle:")
print("*******************************")
npa1 = np.array([1, 2, 3])
npa2 = np.array([4, 5, 6])

print("First argument before task submitting is", npa1)
# without using handle, set option arg_handle to False
res15 = starpu.task_submit(arg_handle=False, ret_handle=True)(np_add, npa1, npa2)
print("First argument after task submitting is", npa1)
#print("The addition result is", res15.get())

res15.unregister()

#########################

starpu.shutdown()
