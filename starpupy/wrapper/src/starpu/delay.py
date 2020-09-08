from starpu import task
def delayed(f):
	def submit():
		task.task_submit(f)
	return submit