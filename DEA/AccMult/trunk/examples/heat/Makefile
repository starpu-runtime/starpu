RIBSCHED= ../../ribsched.a
CFLAGS+= -I../..

ifdef OPENGL_RENDER
	CFLAGS+=-DOPENGL_RENDER
	LDFLAGS+=-lglut -lGL -lGLU
endif


all: heat

%.o: %.c
	$(CC) $(CFLAGS) -c -o $@ $<

heat: dw_sparse_cg.o dw_sparse_cg_kernels.o dw_factolu.o dw_factolu_tag.o heat_display.o heat.o lu_kernels_model.o dw_factolu_kernels.o
	$(CC) dw_sparse_cg.o dw_sparse_cg_kernels.o dw_factolu.o dw_factolu_tag.o heat_display.o heat.o lu_kernels_model.o dw_factolu_kernels.o -o heat $(LDFLAGS) $(RIBSCHED)

clean:
	@rm -f *.o *.d
	@rm -f heat
