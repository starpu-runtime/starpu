PYTHON ?= python3.6

CPPFLAGS = $(shell $(PYTHON)-config --includes) -Wall -O2 -g
CFLAGS += $(shell pkg-config --cflags starpu-1.3)
LDLIBS += $(shell pkg-config --libs starpu-1.3)

all: task.so

task.so: starpu_task_wrapper.c Makefile
	$(CC) -fPIC $(CFLAGS) $< -o $@ -shared  $(CPPFLAGS) $(LDLIBS)

clean:
	rm -f task.so
