from starpu import task
import asyncio

def delayed(f):
	def submit():
		async def fut_wait():
			fut = task.task_submit(f)
			await fut
		asyncio.run(fut_wait())
	return submit
