import starpu
import time

@starpu.delayed
def salut():
	time.sleep(1)
	print ("salut, le monde")


def hello():
	print ("print in python")
	print ("Hello, world!")

starpu.pause()

print("begin to submit task in python")
fut=starpu.task_submit(hello)
salut()
#starpu.task_submit(hello)
#starpu.task_submit(hello)
print("finish to submit task in python")
starpu.resume()

#print("begin to sleep")
#time.sleep(1)
#print("finish to sleep")

starpu.task_wait_for_all()