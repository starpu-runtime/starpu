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
    fut = starpu.task_submit(hello)
    await fut
asyncio.run(hello_wait())

#############################################################################

#function no input no output
def func1():
	print ("Example 2:")
	print ("This is a function no input no output")

#submit function "func1"
async def func1_wait():
    fut1 = starpu.task_submit(func1)
    await fut1
asyncio.run(func1_wait())

##############################################################################

#using decorator wrap the function
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
    fut2 = starpu.task_submit(func2)
    res = await fut2
    #print the result of function
    print("This is a function no input and the return value is", res)
asyncio.run(func2_wait())

###############################################################################

starpu.task_wait_for_all()