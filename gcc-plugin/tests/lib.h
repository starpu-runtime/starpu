/* GCC-StarPU
   Copyright (C) 2011 Institut National de Recherche en Informatique et Automatique

   GCC-StarPU is free software: you can redistribute it and/or modify
   it under the terms of the GNU General Public License as published by
   the Free Software Foundation, either version 3 of the License, or
   (at your option) any later version.

   GCC-StarPU is distributed in the hope that it will be useful,
   but WITHOUT ANY WARRANTY; without even the implied warranty of
   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
   GNU General Public License for more details.

   You should have received a copy of the GNU General Public License
   along with GCC-StarPU.  If not, see <http://www.gnu.org/licenses/>.  */

/* Testing library, including stubs of StarPU functions.  */

#ifndef STARPU_GCC_PLUGIN
# error barf!
#endif

#ifndef STARPU_USE_CPU
# error damn it!
#endif

#undef NDEBUG

#include <stdlib.h>
#include <stdarg.h>
#include <string.h>
#include <assert.h>


/* Stub used for testing purposes.  */

/* Number of tasks submitted.  */
static unsigned int tasks_submitted;

struct insert_task_argument
{
  /* `STARPU_VALUE', etc. */
  int type;

  /* Pointer to the expected value.  */
  const void *pointer;

  /* Size in bytes of the data pointed to.  */
  size_t size;
};

/* Pointer to a zero-terminated array listing the expected
   `starpu_insert_task' arguments.  */
const struct insert_task_argument *expected_insert_task_arguments;

int
starpu_insert_task (starpu_codelet *cl, ...)
{
  assert (cl->where == (STARPU_CPU | STARPU_OPENCL));

  /* TODO: Call `cpu_func' & co. and check whether they do the right
     thing.  */

  assert (cl->cpu_func != NULL);
  assert (cl->opencl_func != NULL);
  assert (cl->cuda_func == NULL);

  va_list args;

  va_start (args, cl);

  const struct insert_task_argument *expected;
  for (expected = expected_insert_task_arguments;
       expected->type != 0;
       expected++)
    {
      int type;

      type = va_arg (args, int);
      assert (type == expected->type);

      switch (type)
	{
	case STARPU_VALUE:
	  {
	    void *arg;
	    size_t size;

	    arg = va_arg (args, void *);
	    size = va_arg (args, size_t);

	    assert (size == expected->size);
	    assert (arg != NULL);
	    assert (!memcmp (arg, expected->pointer, size));
	    break;
	  }

	case STARPU_RW:
	case STARPU_R:
	case STARPU_W:
	  {
	    starpu_data_handle handle;
	    handle = starpu_data_lookup (expected->pointer);
	    assert (va_arg (args, void *) == handle);
	    break;
	  }

	default:
	  abort ();
	}
    }

  va_end (args);

  /* Make sure all the arguments were consumed.  */
  assert (expected->type == 0);

  tasks_submitted++;

  return 0;
}

/* Our own implementation of `starpu_unpack_cl_args', for debugging
   purposes.  */

void
starpu_unpack_cl_args (void *cl_raw_arg, ...)
{
  va_list args;
  size_t nargs, arg, offset, size;
  unsigned char *cl_arg;

  cl_arg = (unsigned char *) cl_raw_arg;

  nargs = *cl_arg;

  va_start (args, cl_raw_arg);

  for (arg = 0, offset = 1, size = 0;
       arg < nargs;
       arg++, offset += sizeof (size_t) + size)
    {
      void *argp;

      argp = va_arg (args, void *);
      size = *(size_t *) &cl_arg[size];

      memcpy (argp, &cl_arg[offset], size);
    }

  va_end (args);
}


/* Data handles.  For testing purposes, there's a dummy implementation of
   data handles below, which disguises the original pointer to form a pseudo
   handle.  This allows us to test whether the task implementation is
   actually passed a pointer, not a handle.  */

#define pointer_as_int(p) ((uintptr_t) (p))
#define int_as_pointer(i) ((void *) (i))

#define dummy_pointer_to_handle(p)		\
  ({						\
     assert ((pointer_as_int (p) & 1) == 0);	\
     int_as_pointer (~pointer_as_int (p));	\
   })

#define dummy_handle_to_pointer(h)		\
  ({						\
     assert ((pointer_as_int (h) & 1) == 1);	\
     int_as_pointer (~pointer_as_int (h));	\
   })

starpu_data_handle
starpu_data_lookup (const void *ptr)
{
  return dummy_pointer_to_handle (ptr);
}

void *
starpu_handle_get_local_ptr (starpu_data_handle handle)
{
  return dummy_handle_to_pointer (handle);
}


/* Data registration.  */

struct data_register_arguments
{
  /* A pointer to the vector being registered.  */
  void *pointer;

  /* Number of elements in the vector.  */
  size_t elements;

  /* Size of individual elements.  */
  size_t element_size;
};

/* Number of `starpu_vector_data_register' calls.  */
static unsigned int data_register_calls;

/* Variable describing the expected `starpu_vector_data_register'
   arguments.  */
struct data_register_arguments expected_register_arguments;

void
starpu_vector_data_register (starpu_data_handle *handle,
			     uint32_t home_node, uintptr_t ptr,
			     uint32_t count, size_t elemsize)
{
  assert ((void *) ptr == expected_register_arguments.pointer);
  assert (count == expected_register_arguments.elements);
  assert (elemsize == expected_register_arguments.element_size);

  data_register_calls++;
  *handle = dummy_pointer_to_handle ((void *) ptr);
}


/* Initialization.  */

static int initialized;

int
starpu_init (struct starpu_conf *config)
{
  initialized++;
  return 0;
}
