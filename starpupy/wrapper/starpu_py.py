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
#starpu.runfunc(hello)
#hello()
print("begin to submit task in python")
fut=starpu.submit(hello)
salut()
#starpu.submit(salut)
#starpu.submit(hello)
#starpu.submit(salut)
#starpu.submit(hello)
print("finish to submit task in python")
starpu.resume()

#starpu.wait(t)
#print("begin to sleep")
#time.sleep(1)
#print("finish to sleep")

starpu.wait_for_all()