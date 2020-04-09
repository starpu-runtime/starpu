SRCS=src/jlstarpu_task_submit.c src/jlstarpu_simple_functions.c src/jlstarpu_data_handles.c
CC = gcc
CFLAGS += $(shell pkg-config --cflags starpu-1.3)
LDFLAGS += $(shell pkg-config --libs starpu-1.3)
# for Mac, using .dylib
lib/libjlstarpu_c_wrapper.dylib: ${SRCS}
	$(CC) -O3 -shared -fPIC $(CFLAGS) $^ -o $@ $(LDFLAGS)
