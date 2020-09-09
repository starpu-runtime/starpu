from starpu import task
import asyncio

def delayed(f):
	def submit(*args,**kwargs):
		async def fut_wait():
			fut = task.task_submit(f, list(args))
			res = await fut
		asyncio.run(fut_wait())
	return submit
