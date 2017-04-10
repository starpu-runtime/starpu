STARPU=../../
CPPFLAGS=-I$(STARPU)/src
CFLAGS=-Wall -Wextra -g $(STARPU)/src/common/rbtree.c
LDFLAGS=-lsimgrid

MC_FLAGS=--cfg=model-check/reduction:none

test: prio_list
	simgrid-mc ./prio_list platform.xml MAIN $(MC_FLAGS)

debug: prio_list
	simgrid-mc ./prio_list platform.xml MAIN --log=mc_safety.thres:debug $(MC_FLAGS)
