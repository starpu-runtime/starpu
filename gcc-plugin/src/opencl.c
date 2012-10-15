/* GCC-StarPU
   Copyright (C) 2012 Inria

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

#include <starpu-gcc/config.h>

#include <stdlib.h>
#include <unistd.h>

#include <starpu.h>

#include <gcc-plugin.h>
#include <plugin-version.h>
#include <plugin.h>
#include <tree.h>
#include <tree-iterator.h>
#include <gimple.h>
#include <cgraph.h>
#include <toplev.h>
#include <langhooks.h>

#ifdef HAVE_C_FAMILY_C_COMMON_H
# include <c-family/c-common.h>
#elif HAVE_C_COMMON_H
# include <c-common.h>
#endif

#include <starpu-gcc/utils.h>
#include <starpu-gcc/tasks.h>


/* Search path for OpenCL source files for the `opencl' pragma, as a
   `TREE_LIST'.  */
tree opencl_include_dirs = NULL_TREE;

/* Names of data structures defined in <starpu.h>.  */
static const char opencl_program_struct_tag[] = "starpu_opencl_program";


/* Return the type corresponding to OPENCL_PROGRAM_STRUCT_TAG.  */

static tree
opencl_program_type (void)
{
  tree t = TREE_TYPE (type_decl_for_struct_tag (opencl_program_struct_tag));

  if (TYPE_SIZE (t) == NULL_TREE)
    {
      /* Incomplete type definition, for instance because <starpu_opencl.h>
	 wasn't included.  */
      error_at (UNKNOWN_LOCATION, "StarPU OpenCL support is lacking");
      t = error_mark_node;
    }

  return t;
}

static tree
opencl_kernel_type (void)
{
  tree t = lookup_name (get_identifier ("cl_kernel"));
  gcc_assert (t != NULL_TREE);
  if (TREE_CODE (t) == TYPE_DECL)
    t = TREE_TYPE (t);
  gcc_assert (TYPE_P (t));
  return t;
}

static tree
opencl_command_queue_type (void)
{
  tree t = lookup_name (get_identifier ("cl_command_queue"));
  gcc_assert (t != NULL_TREE);
  if (TREE_CODE (t) == TYPE_DECL)
    t = TREE_TYPE (t);
  gcc_assert (TYPE_P (t));
  return t;
}

static tree
opencl_event_type (void)
{
  tree t = lookup_name (get_identifier ("cl_event"));
  gcc_assert (t != NULL_TREE);
  if (TREE_CODE (t) == TYPE_DECL)
    t = TREE_TYPE (t);
  gcc_assert (TYPE_P (t));
  return t;
}



/* Return a private global string literal VAR_DECL, whose contents are the
   LEN bytes at CONTENTS.  */

static tree
build_string_variable (location_t loc, const char *name_seed,
		       const char *contents, size_t len)
{
  tree decl;

  decl = build_decl (loc, VAR_DECL, create_tmp_var_name (name_seed),
		     string_type_node);
  TREE_PUBLIC (decl) = false;
  TREE_STATIC (decl) = true;
  TREE_USED (decl) = true;

  DECL_INITIAL (decl) =				  /* XXX: off-by-one? */
    build_string_literal (len + 1, contents);

  DECL_ARTIFICIAL (decl) = true;

  return decl;
}

/* Return a VAR_DECL for a string variable containing the contents of FILE,
   which is looked for in each of the directories listed in SEARCH_PATH.  If
   FILE could not be found, return NULL_TREE.  */

static tree
build_variable_from_file_contents (location_t loc,
				   const char *name_seed,
				   const char *file,
				   const_tree search_path)
{
  gcc_assert (search_path != NULL_TREE
	      && TREE_CODE (search_path) == TREE_LIST);

  int err, dir_fd;
  struct stat st;
  const_tree dirs;
  tree var = NULL_TREE;

  /* Look for FILE in each directory in SEARCH_PATH, and pick the first one
     that matches.  */
  for (err = ENOENT, dir_fd = -1, dirs = search_path;
       (err != 0 || err == ENOENT) && dirs != NULL_TREE;
       dirs = TREE_CHAIN (dirs))
    {
      gcc_assert (TREE_VALUE (dirs) != NULL_TREE
		  && TREE_CODE (TREE_VALUE (dirs)) == STRING_CST);

      dir_fd = open (TREE_STRING_POINTER (TREE_VALUE (dirs)),
		     O_DIRECTORY | O_RDONLY);
      if (dir_fd < 0)
	err = ENOENT;
      else
	{
	  err = fstatat (dir_fd, file, &st, 0);
	  if (err != 0)
	    close (dir_fd);
	  else
	    /* Leave DIRS unchanged so it can be referred to in diagnostics
	       below.  */
	    break;
	}
    }

  if (err != 0 || dir_fd < 0)
    error_at (loc, "failed to access %qs: %m", file);
  else if (st.st_size == 0)
    {
      error_at (loc, "source file %qs is empty", file);
      close (dir_fd);
    }
  else
    {
      if (verbose_output_p)
	inform (loc, "found file %qs in %qs",
		file, TREE_STRING_POINTER (TREE_VALUE (dirs)));

      int fd;

      fd = openat (dir_fd, file, O_RDONLY);
      close (dir_fd);

      if (fd < 0)
	error_at (loc, "failed to open %qs: %m", file);
      else
	{
	  void *contents;

	  contents = mmap (NULL, st.st_size, PROT_READ, MAP_SHARED, fd, 0);
	  if (contents == NULL)
	    error_at (loc, "failed to map contents of %qs: %m", file);
	  else
	    {
	      var = build_string_variable (loc, name_seed,
					   (char *) contents, st.st_size);
	      pushdecl (var);
	      munmap (contents, st.st_size);
	    }

	  close (fd);
	}
    }

  return var;
}

/* Return an expression that, given the OpenCL error code in ERROR_VAR,
   returns a string.  */

static tree
build_opencl_error_string (tree error_var)
{
  static tree clstrerror_fn;
  LOOKUP_STARPU_FUNCTION (clstrerror_fn, "starpu_opencl_error_string");

  return build_call_expr (clstrerror_fn, 1, error_var);
}

/* Return an error-checking `clSetKernelArg' call for argument ARG, at
   index IDX, of KERNEL.  */

static tree
build_opencl_set_kernel_arg_call (location_t loc, tree fn,
				  tree kernel, unsigned int idx,
				  tree arg)
{
  gcc_assert (TREE_CODE (fn) == FUNCTION_DECL
	      && TREE_TYPE (kernel) == opencl_kernel_type ());

  static tree setkernarg_fn;
  LOOKUP_STARPU_FUNCTION (setkernarg_fn, "clSetKernelArg");

  tree call = build_call_expr (setkernarg_fn, 4, kernel,
			       build_int_cst (integer_type_node, idx),
			       size_in_bytes (TREE_TYPE (arg)),
			       build_addr (arg, fn));
  tree error_var = build_decl (loc, VAR_DECL,
			       create_tmp_var_name ("setkernelarg_error"),
			       integer_type_node);
  DECL_ARTIFICIAL (error_var) = true;
  DECL_CONTEXT (error_var) = fn;

  tree assignment = build2 (INIT_EXPR, TREE_TYPE (error_var),
			    error_var, call);

  /* Build `if (ERROR_VAR != 0) error ();'.  */
  tree cond;
  cond = build3 (COND_EXPR, void_type_node,
		 build2 (NE_EXPR, boolean_type_node,
			 error_var, integer_zero_node),
		 build_error_statements (loc, error_var,
					 build_opencl_error_string,
					 "failed to set OpenCL kernel "
					 "argument %d", idx),
		 NULL_TREE);

  tree stmts = NULL_TREE;
  append_to_statement_list (assignment, &stmts);
  append_to_statement_list (cond, &stmts);

  return build4 (TARGET_EXPR, void_type_node, error_var,
		 stmts, NULL_TREE, NULL_TREE);
}

/* Return the sequence of `clSetKernelArg' calls for KERNEL.  */

static tree
build_opencl_set_kernel_arg_calls (location_t loc, tree task_impl,
				   tree kernel)
{
  gcc_assert (task_implementation_p (task_impl));

  size_t n;
  tree arg, stmts = NULL_TREE;

  for (arg = DECL_ARGUMENTS (task_impl), n = 0;
       arg != NULL_TREE;
       arg = TREE_CHAIN (arg), n++)
    {
      tree call = build_opencl_set_kernel_arg_call (loc, task_impl,
						    kernel, n, arg);
      append_to_statement_list (call, &stmts);
    }

  return stmts;
}

/* Define a body for TASK_IMPL that loads OpenCL source from FILE and calls
   KERNEL.  */

static void
define_opencl_task_implementation (location_t loc, tree task_impl,
				   const char *file, const_tree kernel,
				   tree groupsize)
{
  gcc_assert (task_implementation_p (task_impl)
	      && task_implementation_where (task_impl) == STARPU_OPENCL);
  gcc_assert (TREE_CODE (kernel) == STRING_CST);
  gcc_assert (INTEGRAL_TYPE_P (TREE_TYPE (groupsize)));

  local_define (tree, local_var, (tree type))
  {
    tree var = build_decl (loc, VAR_DECL,
			   create_tmp_var_name ("opencl_var"),
			   type);
    DECL_ARTIFICIAL (var) = true;
    DECL_CONTEXT (var) = task_impl;
    return var;
  };

  if (!verbose_output_p)
    /* No further warnings for this node.  */
    TREE_NO_WARNING (task_impl) = true;

  static tree load_fn, load_kern_fn, enqueue_kern_fn, wid_fn, devid_fn, clfinish_fn,
    collect_stats_fn, release_ev_fn;

  if (load_fn == NULL_TREE)
    {
      load_fn =
	lookup_name (get_identifier ("starpu_opencl_load_opencl_from_string"));
      if (load_fn == NULL_TREE)
	{
	  inform (loc, "no OpenCL support, task implementation %qE "
		  "not generated", DECL_NAME (task_impl));
	  return;
	}
    }

  LOOKUP_STARPU_FUNCTION (load_kern_fn, "starpu_opencl_load_kernel");
  LOOKUP_STARPU_FUNCTION (wid_fn, "starpu_worker_get_id");
  LOOKUP_STARPU_FUNCTION (devid_fn, "starpu_worker_get_devid");
  LOOKUP_STARPU_FUNCTION (enqueue_kern_fn, "clEnqueueNDRangeKernel");
  LOOKUP_STARPU_FUNCTION (clfinish_fn, "clFinish");
  LOOKUP_STARPU_FUNCTION (collect_stats_fn, "starpu_opencl_collect_stats");
  LOOKUP_STARPU_FUNCTION (release_ev_fn, "clReleaseEvent");

  if (verbose_output_p)
    inform (loc, "defining %qE, with OpenCL kernel %qs from file %qs",
	    DECL_NAME (task_impl), TREE_STRING_POINTER (kernel), file);

  tree source_var;
  source_var = build_variable_from_file_contents (loc, "opencl_source",
						  file, opencl_include_dirs);
  if (source_var != NULL_TREE)
    {
      /* Give TASK_IMPL an actual argument list.  */
      DECL_ARGUMENTS (task_impl) = build_function_arguments (task_impl);

      tree prog_var, prog_loaded_var;

      /* Global variable to hold the `starpu_opencl_program' object.  */

      prog_var = build_decl (loc, VAR_DECL,
			     create_tmp_var_name ("opencl_program"),
			     opencl_program_type ());
      TREE_PUBLIC (prog_var) = false;
      TREE_STATIC (prog_var) = true;
      TREE_USED (prog_var) = true;
      DECL_ARTIFICIAL (prog_var) = true;
      pushdecl (prog_var);

      /* Global variable indicating whether the program has already been
	 loaded.  */

      prog_loaded_var = build_decl (loc, VAR_DECL,
				    create_tmp_var_name ("opencl_prog_loaded"),
				    boolean_type_node);
      TREE_PUBLIC (prog_loaded_var) = false;
      TREE_STATIC (prog_loaded_var) = true;
      TREE_USED (prog_loaded_var) = true;
      DECL_ARTIFICIAL (prog_loaded_var) = true;
      DECL_INITIAL (prog_loaded_var) = build_zero_cst (boolean_type_node);
      pushdecl (prog_loaded_var);

      /* Build `starpu_opencl_load_opencl_from_string (SOURCE_VAR,
	                                               &PROG_VAR, "")'.  */
      tree load = build_call_expr (load_fn, 3, source_var,
				   build_addr (prog_var, task_impl),
				   build_string_literal (1, ""));

      tree load_stmts = NULL_TREE;
      append_to_statement_list (load, &load_stmts);
      append_to_statement_list (build2 (MODIFY_EXPR, boolean_type_node,
					prog_loaded_var,
					build_int_cst (boolean_type_node, 1)),
				&load_stmts);

      /* Build `if (!PROG_LOADED_VAR) { ...; PROG_LOADED_VAR = true; }'.  */

      tree load_cond = build3 (COND_EXPR, void_type_node,
			       prog_loaded_var,
			       NULL_TREE,
			       load_stmts);

      /* Local variables.  */
      tree kernel_var, queue_var, event_var, group_size_var, ngroups_var,
	error_var;

      kernel_var = local_var (opencl_kernel_type ());
      queue_var = local_var (opencl_command_queue_type ());
      event_var = local_var (opencl_event_type ());
      group_size_var = local_var (size_type_node);
      ngroups_var = local_var (size_type_node);
      error_var = local_var (integer_type_node);

      /* Build `starpu_opencl_load_kernel (...)'.
         TODO: Check return value.  */
      tree devid =
	build_call_expr (devid_fn, 1, build_call_expr (wid_fn, 0));

      tree load_kern = build_call_expr (load_kern_fn, 5,
					build_addr (kernel_var, task_impl),
					build_addr (queue_var, task_impl),
					build_addr (prog_var, task_impl),
					build_string_literal
					(TREE_STRING_LENGTH (kernel) + 1,
					 TREE_STRING_POINTER (kernel)),
					devid);

      tree enqueue_kern =
	build_call_expr (enqueue_kern_fn, 9,
			 queue_var, kernel_var,
			 build_int_cst (integer_type_node, 1),
			 null_pointer_node,
			 build_addr (group_size_var, task_impl),
			 build_addr (ngroups_var, task_impl),
			 integer_zero_node,
			 null_pointer_node,
			 build_addr (event_var, task_impl));
      tree enqueue_err =
	build2 (INIT_EXPR, TREE_TYPE (error_var), error_var, enqueue_kern);

      tree enqueue_cond =
	build3 (COND_EXPR, void_type_node,
		build2 (NE_EXPR, boolean_type_node,
			error_var, integer_zero_node),
		build_error_statements (loc, error_var,
					build_opencl_error_string,
					"failed to enqueue kernel"),
		NULL_TREE);

      tree clfinish =
	build_call_expr (clfinish_fn, 1, queue_var);

      tree collect_stats =
	build_call_expr (collect_stats_fn, 1, event_var);

      tree release_ev =
	build_call_expr (release_ev_fn, 1, event_var);

      tree enqueue_stmts = NULL_TREE;
      append_to_statement_list (enqueue_err, &enqueue_stmts);
      append_to_statement_list (enqueue_cond, &enqueue_stmts);


      /* TODO: Build `clFinish', `clReleaseEvent', & co.  */
      /* Put it all together.  */
      tree stmts = NULL_TREE;
      append_to_statement_list (load_cond, &stmts);
      append_to_statement_list (load_kern, &stmts);
      append_to_statement_list (build_opencl_set_kernel_arg_calls (loc,
								   task_impl,
								   kernel_var),
				&stmts);

      /* TODO: Support user-provided values.  */
      append_to_statement_list (build2 (INIT_EXPR, TREE_TYPE (group_size_var),
					group_size_var,
					fold_convert (TREE_TYPE (group_size_var),
						      groupsize)),
				&stmts);
      append_to_statement_list (build2 (INIT_EXPR, TREE_TYPE (ngroups_var),
					ngroups_var,
					build_int_cst (TREE_TYPE (ngroups_var),
						       1)),
				&stmts);
      append_to_statement_list (build4 (TARGET_EXPR, void_type_node,
					error_var, enqueue_stmts,
					NULL_TREE, NULL_TREE),
				&stmts);
      append_to_statement_list (clfinish, &stmts);
      append_to_statement_list (collect_stats, &stmts);
      append_to_statement_list (release_ev, &stmts);

      /* Bind the local vars.  */
      tree vars = chain_trees (kernel_var, queue_var, event_var,
			       group_size_var, ngroups_var, NULL_TREE);
      tree bind = build3 (BIND_EXPR, void_type_node, vars, stmts,
			  build_block (vars, NULL_TREE, task_impl, NULL_TREE));

      TREE_USED (task_impl) = true;
      TREE_STATIC (task_impl) = true;
      DECL_EXTERNAL (task_impl) = false;
      DECL_ARTIFICIAL (task_impl) = true;
      DECL_SAVED_TREE (task_impl) = bind;
      DECL_INITIAL (task_impl) = BIND_EXPR_BLOCK (bind);
      DECL_RESULT (task_impl) =
	build_decl (loc, RESULT_DECL, NULL_TREE, void_type_node);

      /* Compile TASK_IMPL.  */
      rest_of_decl_compilation (task_impl, true, 0);
      allocate_struct_function (task_impl, false);
      cgraph_finalize_function (task_impl, false);
      cgraph_mark_needed_node (cgraph_get_node (task_impl));

      /* Generate a wrapper for TASK_IMPL, and possibly the body of its task.
	 This needs to be done explicitly here, because otherwise
	 `handle_pre_genericize' would never see TASK_IMPL's task.  */
      tree task = task_implementation_task (task_impl);
      if (!TREE_STATIC (task))
	{
	  declare_codelet (task);
	  define_task (task);

	  /* Compile TASK's body.  */
	  rest_of_decl_compilation (task, true, 0);
	  allocate_struct_function (task, false);
	  cgraph_finalize_function (task, false);
	  cgraph_mark_needed_node (cgraph_get_node (task));
	}
    }
  else
    DECL_SAVED_TREE (task_impl) = error_mark_node;

  return;
}

/* Handle the `opencl' pragma, which defines an OpenCL task
   implementation.  */

void
handle_pragma_opencl (struct cpp_reader *reader)
{
  tree args;
  location_t loc;

  loc = cpp_peek_token (reader, 0)->src_loc;

  if (current_function_decl != NULL_TREE)
    {
      error_at (loc, "%<starpu opencl%> pragma can only be used "
		"at the top-level");
      return;
    }

  args = read_pragma_expressions ("opencl", loc);
  if (args == NULL_TREE)
    return;

  /* TODO: Add "number of groups" arguments.  */
  if (list_length (args) < 4)
    {
      error_at (loc, "wrong number of arguments for %<starpu opencl%> pragma");
      return;
    }

  if (task_implementation_p (TREE_VALUE (args)))
    {
      tree task_impl = TREE_VALUE (args);
      if (task_implementation_where (task_impl) == STARPU_OPENCL)
  	{
  	  args = TREE_CHAIN (args);
  	  if (TREE_CODE (TREE_VALUE (args)) == STRING_CST)
  	    {
  	      tree file = TREE_VALUE (args);
  	      args = TREE_CHAIN (args);
  	      if (TREE_CODE (TREE_VALUE (args)) == STRING_CST)
  		{
  		  tree kernel = TREE_VALUE (args);
		  args = TREE_CHAIN (args);

		  if (TREE_TYPE (TREE_VALUE (args)) != NULL_TREE &&
		      INTEGRAL_TYPE_P (TREE_TYPE (TREE_VALUE (args))))
		    {
		      tree groupsize = TREE_VALUE (args);
		      if (TREE_CHAIN (args) == NULL_TREE)
			define_opencl_task_implementation (loc, task_impl,
							   TREE_STRING_POINTER (file),
							   kernel, groupsize);
		      else
			error_at (loc, "junk after %<starpu opencl%> pragma");
		    }
		  else
		    error_at (loc, "%<groupsize%> argument must be an integral type");
  		}
  	      else
  		error_at (loc, "%<kernel%> argument must be a string constant");
	    }
	  else
	    error_at (loc, "%<file%> argument must be a string constant");
	}
      else
	error_at (loc, "%qE is not an OpenCL task implementation",
		  DECL_NAME (task_impl));
    }
  else
    error_at (loc, "%qE is not a task implementation", TREE_VALUE (args));
}

/* Diagnose use of C types that are either nonexistent or different in
   OpenCL.  */

void
validate_opencl_argument_type (location_t loc, const_tree type)
{
  /* When TYPE is a pointer type, get to the base element type.  */
  for (; POINTER_TYPE_P (type); type = TREE_TYPE (type));

  if (!RECORD_OR_UNION_TYPE_P (type) && !VOID_TYPE_P (type))
    {
      tree decl = TYPE_NAME (type);

      if (DECL_P (decl))
	{
	  static const struct { const char *c; const char *cl; }
	  type_map[] =
	    {
	      /* Scalar types defined in OpenCL 1.2.  See
		 <http://www.khronos.org/files/opencl-1-2-quick-reference-card.pdf>.  */
	      { "char", "cl_char" },
	      { "signed char", "cl_char" },
	      { "unsigned char", "cl_uchar" },
	      { "uchar", "cl_uchar" },
	      { "short int", "cl_short" },
	      { "unsigned short", "cl_ushort" },
	      { "int", "cl_int" },
	      { "unsigned int", "cl_uint" },
	      { "uint", "cl_uint" },
	      { "long int", "cl_long" },
	      { "long unsigned int", "cl_ulong" },
	      { "ulong", "cl_ulong" },
	      { "float", "cl_float" },
	      { "double", "cl_double" },
	      { NULL, NULL }
	    };

	  const char *c_name = IDENTIFIER_POINTER (DECL_NAME (decl));
	  const char *cl_name =
	    ({
	      size_t i;
	      for (i = 0; type_map[i].c != NULL; i++)
		{
		  if (strcmp (type_map[i].c, c_name) == 0)
		    break;
		}
	      type_map[i].cl;
	    });

	  if (cl_name != NULL)
	    {
	      tree cl_type = lookup_name (get_identifier (cl_name));

	      if (cl_type != NULL_TREE)
		{
		  if (DECL_P (cl_type))
		    cl_type = TREE_TYPE (cl_type);

		  if (!lang_hooks.types_compatible_p ((tree) type, cl_type))
		    {
		      tree st, sclt;

		      st = c_common_signed_type ((tree) type);
		      sclt = c_common_signed_type (cl_type);

		      if (st == sclt)
			warning_at (loc, 0, "C type %qE differs in signedness "
				    "from the same-named OpenCL type",
				    DECL_NAME (decl));
		      else
			/* TYPE should be avoided because the it differs from
			   CL_TYPE, and thus cannot be used safely in
			   `clSetKernelArg'.  */
			warning_at (loc, 0, "C type %qE differs from the "
				    "same-named OpenCL type",
				    DECL_NAME (decl));
		    }
		}

	      /* Otherwise we can't conclude.  It could be that <CL/cl.h>
		 wasn't included in the program, for instance.  */
	    }
	  else
	    /* Recommend against use of `size_t', etc.  */
	    warning_at (loc, 0, "%qE does not correspond to a known "
			"OpenCL type", DECL_NAME (decl));
	}
    }
}
