STARPU=../../
CPPFLAGS=-I$(STARPU)/src
CFLAGS=-Wall -Wextra -g $(STARPU)/src/common/rbtree.c
LDFLAGS=-lsimgrid

test: prio_list
	simgrid-mc ./prio_list platform.xml MAIN

debug: prio_list
	simgrid-mc ./prio_list platform.xml MAIN --log=mc_safety.thres:debug
