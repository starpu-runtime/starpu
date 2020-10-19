PYTHON ?= python3

all:
	PYTHONPATH=../src $(PYTHON) starpu_py.py
	PYTHONPATH=../src STARPU_CALIBRATE=1 $(PYTHON) starpu_py_parallel.py

