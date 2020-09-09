import starpu
import time
import asyncio

############################################################################
#function no input no output print hello world
def hello():
	print ("Example 1:")
	print ("Hello, world!")

#submit function "hello"
async def hello_wait():
    fut = starpu.task_submit(hello,[])
    await fut
asyncio.run(hello_wait())

#############################################################################

#function no input no output
def func1():
	print ("Example 2:")
	print ("This is a function no input no output")

#submit function "func1"
async def func1_wait():
    fut1 = starpu.task_submit(func1,[])
    await fut1
asyncio.run(func1_wait())

##############################################################################

#using decorator wrap the function no input no output
@starpu.delayed
def func1_deco():
	#time.sleep(1)
	print ("Example 3:")
	print ("This is a function wrapped by the decorator function")

#apply starpu.delayed(func1_deco())
func1_deco()

##############################################################################

#function no input return a value
def func2():
	print ("Example 4:")
	return 12

#submit function "func2"
async def func2_wait():
    fut2 = starpu.task_submit(func2, [])
    res2 = await fut2
    #print the result of function
    print("This is a function no input and the return value is", res2)
asyncio.run(func2_wait())

###############################################################################
 
#function has 2 int inputs and 1 int output
def multi(a,b):
	print ("Example 5:")
	return a*b

#submit function "multi"
async def multi_wait():
	fut3 = starpu.task_submit(multi, [2, 3])
	res3=await fut3
	print("The result of function multi is :", res3)
asyncio.run(multi_wait())
#print(multi(2, 3))

###############################################################################

#function has 4 float inputs and 1 float output
def add(a,b,c,d):
	print ("Example 6:")
	return a+b+c+d

#submit function "add"
async def add_wait():
	fut4 = starpu.task_submit(add, [1.2, 2.5, 3.6, 4.9])
	res4=await fut4
	print("The result of function add is :", res4)
asyncio.run(add_wait())
#print(add(1.2, 2.5, 3.6, 4.9))

###############################################################################

#function has 2 int inputs 1 float input and 1 float output 1 int output
def sub(a,b,c):
	print ("Example 7:")
	return a-b-c, a-b

#submit function "sub"
async def sub_wait():
	fut5 = starpu.task_submit(sub, [6, 2, 5.9])
	res5 = await fut5
	print("The result of function sub is:", res5)
asyncio.run(sub_wait())
#print(sub(6, 2, 5.9))

###############################################################################

starpu.task_wait_for_all()