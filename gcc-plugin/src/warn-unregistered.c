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

/* Use extensions of the GNU C Library.  */
#define _GNU_SOURCE 1

#include <starpu-gcc-config.h>

#include <gcc-plugin.h>
#include <plugin-version.h>

#include <plugin.h>
#include <cpplib.h>
#include <tree.h>
#include <tree-pass.h>
#include <gimple.h>
#include <diagnostic.h>
#include <cgraph.h>

#include <utils.h>
#include <tasks.h>

/* Return true if there exists a `starpu_vector_data_register' call for VAR
   before GSI in its basic block.  */

static bool
registration_in_bb_p (gimple_stmt_iterator gsi, tree var)
{
  gcc_assert (SSA_VAR_P (var));

  tree register_fn_name;

  register_fn_name = get_identifier ("starpu_vector_data_register");

  local_define (bool, registration_function_p, (const_tree obj))
  {
    /* TODO: Compare against the real fndecl.  */
    return (obj != NULL_TREE
	    && TREE_CODE (obj) == FUNCTION_DECL
	    && DECL_NAME (obj) == register_fn_name);
  };

  bool found;

  for (found = false;
       !gsi_end_p (gsi) && !found;
       gsi_prev (&gsi))
    {
      gimple stmt;

      stmt = gsi_stmt (gsi);
      if (is_gimple_call (stmt))
	{
	  tree fn = gimple_call_fndecl (stmt);
	  if (registration_function_p (fn))
	    {
	      tree arg = gimple_call_arg (stmt, 2);
	      if (is_gimple_address (arg))
	      	arg = TREE_OPERAND (arg, 0);

	      if (((TREE_CODE (arg) == VAR_DECL
		    || TREE_CODE (arg) == VAR_DECL)
		   && refs_may_alias_p (arg, var))

		  /* Both VAR and ARG should be SSA names, otherwise, if ARG
		     is a VAR_DECL, `ptr_derefs_may_alias_p' will
		     conservatively assume that they may alias.  */
		  || (TREE_CODE (var) == SSA_NAME
		      && TREE_CODE (arg) != VAR_DECL
		      && ptr_derefs_may_alias_p (arg, var)))
		{
		  if (verbose_output_p)
		    {
		      var = TREE_CODE (var) == SSA_NAME ? SSA_NAME_VAR (var) : var;
		      inform (gimple_location (stmt),
			      "found registration of variable %qE",
			      DECL_NAME (var));
		    }
		  found = true;
		}
	    }
	}
    }

  return found;
}

/* Return true if BB is dominated by a registration of VAR.  */

static bool
dominated_by_registration (gimple_stmt_iterator gsi, tree var)
{
  /* Is there a registration call for VAR in GSI's basic block?  */
  if (registration_in_bb_p (gsi, var))
    return true;

  edge e;
  edge_iterator ei;
  bool found = false;

  /* If every incoming edge is dominated by a registration, then we're
     fine.

     FIXME: This triggers false positives when registration is done in a
     loop, because there's always an edge through which no registration
     happens--the edge corresponding to the case where the loop is not
     entered.  */

  FOR_EACH_EDGE (e, ei, gsi_bb (gsi)->preds)
    {
      if (!dominated_by_registration (gsi_last_bb (e->src), var))
	return false;
      else
	found = true;
    }

  return found;
}

/* Return true if NAME aliases a global variable or a PARM_DECL.  Note that,
   for the former, `ptr_deref_may_alias_global_p' is way too conservative,
   hence this approach.  */

static bool
ssa_name_aliases_global_or_parm_p (const_tree name)
{
  gcc_assert (TREE_CODE (name) == SSA_NAME);

  if (TREE_CODE (SSA_NAME_VAR (name)) == PARM_DECL)
    return true;
  else
    {
      gimple def_stmt;

      def_stmt = SSA_NAME_DEF_STMT (name);
      if (is_gimple_assign (def_stmt))
	{
	  tree rhs = gimple_assign_rhs1 (def_stmt);

	  if (TREE_CODE (rhs) == VAR_DECL
	      && (DECL_EXTERNAL (rhs) || TREE_STATIC (rhs)))
	    return true;
	}
    }

  return false;
}


/* Validate the arguments passed to tasks in FN's body.  */

static void
validate_task_invocations (tree fn)
{
  gcc_assert (TREE_CODE (fn) == FUNCTION_DECL);

  const struct cgraph_node *cgraph;
  const struct cgraph_edge *callee;

  cgraph = cgraph_get_node (fn);

  /* When a definition of IMPL is available, check its callees.  */
  if (cgraph != NULL)
    for (callee = cgraph->callees;
	 callee != NULL;
	 callee = callee->next_callee)
      {
	if (task_p (callee->callee->decl))
	  {
	    unsigned i;
	    gimple call_stmt = callee->call_stmt;

	    for (i = 0; i < gimple_call_num_args (call_stmt); i++)
	      {
		tree arg = gimple_call_arg (call_stmt, i);

		if (TREE_CODE (arg) == ADDR_EXPR
		    && TREE_CODE (TREE_OPERAND (arg, 0)) == VAR_DECL
		    && (TREE_CODE (TREE_TYPE (TREE_OPERAND (arg, 0)))
			== ARRAY_TYPE))
		  /* This is a "pointer-to-array" of a variable, so what we
		     really care about is the variable itself.  */
		  arg = TREE_OPERAND (arg, 0);

		if ((POINTER_TYPE_P (TREE_TYPE (arg))
		     || (TREE_CODE (TREE_TYPE (arg)) == ARRAY_TYPE))
		    && ((TREE_CODE (arg) == VAR_DECL
			 && !TREE_STATIC (arg)
			 && !DECL_EXTERNAL (arg)
			 && !TREE_NO_WARNING (arg))
			|| (TREE_CODE (arg) == SSA_NAME
			    && !ssa_name_aliases_global_or_parm_p (arg))))
		  {
		    if (!dominated_by_registration (gsi_for_stmt (call_stmt),
						    arg))
		      {
			if (TREE_CODE (arg) == SSA_NAME)
			  {
			    tree var = SSA_NAME_VAR (arg);
			    if (DECL_NAME (var) != NULL)
			      arg = var;

			    /* TODO: Check whether we can get the original
			       variable name via ARG's DEF_STMT.  */
			  }

			if (TREE_CODE (arg) == VAR_DECL
			    && DECL_NAME (arg) != NULL_TREE)
			  warning_at (gimple_location (call_stmt), 0,
				      "variable %qE may be used unregistered",
				      DECL_NAME (arg));
			else
			  warning_at (gimple_location (call_stmt), 0,
				      "argument %i may be used unregistered",
				      i);
		      }
		  }
	      }
	  }
      }
}

/* A pass to warn about possibly unregistered task arguments.  */

static unsigned int
warn_starpu_unregistered (void)
{
  tree fndecl;

  fndecl = current_function_decl;
  gcc_assert (TREE_CODE (fndecl) == FUNCTION_DECL);

  if (!task_p (fndecl))
    validate_task_invocations (fndecl);

  return 0;
}

struct opt_pass pass_warn_starpu_unregistered =
  {
    designated_field_init (type, GIMPLE_PASS),
    designated_field_init (name, "warn_starpu_unregistered"),
    designated_field_init (gate, NULL),
    designated_field_init (execute, warn_starpu_unregistered),

    /* The rest is zeroed.  */
  };
