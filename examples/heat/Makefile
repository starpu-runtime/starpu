STARPU= ../../starpu.a
CFLAGS+= -I../../src/

OBJS := dw_sparse_cg.o dw_sparse_cg_kernels.o dw_factolu.o dw_factolu_tag.o heat_display.o heat.o lu_kernels_model.o dw_factolu_kernels.o

ifdef OPENGL_RENDER
	CFLAGS+=-DOPENGL_RENDER
	LDFLAGS+=-lglut -lGL -lGLU
endif

all: heat

%.d: %.c
	$(CC) $(CFLAGS) $< -MM -o $*.d

-include $(OBJS:.o=.d)

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

heat: $(STARPU) $(OBJS) 
	$(CC) $(OBJS) -o heat $(LDFLAGS) $(STARPU) $(EXTRALDFLAGS)

clean:
	@rm -f *.o *.d *.gcno *.gcda
	@rm -f heat
