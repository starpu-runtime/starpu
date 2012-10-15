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

#include <starpu-gcc-config.h>

#include <gcc-plugin.h>
#include <plugin-version.h>

#include <plugin.h>
#include <cpplib.h>
#include <tree.h>
#include <gimple.h>

#include <utils.h>


/* Whether to enable verbose output.  */
bool verbose_output_p = false;

/* Name of the `task' attribute.  */
const char task_attribute_name[] = "task";

/* Return true if DECL is a task.  */

bool
task_p (const_tree decl)
{
  return (TREE_CODE (decl) == FUNCTION_DECL &&
	  lookup_attribute (task_attribute_name,
			    DECL_ATTRIBUTES (decl)) != NULL_TREE);
}

/* C expression parser, possibly with C++ linkage.  */

extern int yyparse (location_t, const char *, tree *);
extern int yydebug;

/* Parse expressions from the CPP reader for PRAGMA, which is located at LOC.
   Return a TREE_LIST of C expressions.  */

tree
read_pragma_expressions (const char *pragma, location_t loc)
{
  tree expr = NULL_TREE;

  if (yyparse (loc, pragma, &expr))
    /* Parse error or memory exhaustion.  */
    expr = NULL_TREE;

  return expr;
}
