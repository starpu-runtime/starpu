from starpu import task
def delayed(f):
	def submit():
		task.submit(f)
	return submit