################################################################################
# Utilities.cmake - Maven support for cmake
#
# Based on BoostUtilities.cmake from CMake configuration for Boost
# and SociUtilities.cmake from CMake configuration for SOCI
################################################################################
# Copyright (C) 2007 Douglas Gregor <doug.gregor@gmail.com>
# Copyright (C) 2007 Troy Straszheim
# Copyright (C) 2010 Mateusz Loskot <mateusz@loskot.net> 
# Copyright (C) 2012 Gregoire Lejeune <gregoire.lejeune@free.fr>
#
# Distributed under the Boost Software License, Version 1.0.
# See accompanying file http://www.boost.org/LICENSE_1_0.txt
################################################################################

# This utility macro determines whether a particular string value
# occurs within a list of strings:
#
#  list_contains(result string_to_find arg1 arg2 arg3 ... argn)
# 
# This macro sets the variable named by result equal to TRUE if
# string_to_find is found anywhere in the following arguments.
macro(list_contains var value)
  set(${var})
  foreach (value2 ${ARGN})
    if (__${value} STREQUAL __${value2})
      set(${var} TRUE)
    endif (__${value} STREQUAL __${value2})
  endforeach (value2)
endmacro(list_contains)

# The parse_arguments macro will take the arguments of another macro and
# define several variables. The first argument to parse_arguments is a
# prefix to put on all variables it creates. The second argument is a
# list of names, and the third argument is a list of options. Both of
# these lists should be quoted. The rest of parse_arguments are
# arguments from another macro to be parsed.
# 
#     parse_arguments(prefix arg_names options arg1 arg2...) 
# 
# For each item in options, parse_arguments will create a variable with
# that name, prefixed with prefix_. So, for example, if prefix is
# MY_MACRO and options is OPTION1;OPTION2, then parse_arguments will
# create the variables MY_MACRO_OPTION1 and MY_MACRO_OPTION2. These
# variables will be set to true if the option exists in the command line
# or false otherwise.
# 
# For each item in arg_names, parse_arguments will create a variable
# with that name, prefixed with prefix_. Each variable will be filled
# with the arguments that occur after the given arg_name is encountered
# up to the next arg_name or the end of the arguments. All options are
# removed from these lists. parse_arguments also creates a
# prefix_DEFAULT_ARGS variable containing the list of all arguments up
# to the first arg_name encountered.
macro(parse_arguments prefix arg_names option_names)
  set(DEFAULT_ARGS)
  foreach(arg_name ${arg_names})
    set(${prefix}_${arg_name})
  endforeach(arg_name)
  foreach(option ${option_names})
    set(${prefix}_${option} FALSE)
  endforeach(option)

  set(current_arg_name DEFAULT_ARGS)
  set(current_arg_list)
  foreach(arg ${ARGN})
    list_contains(is_arg_name ${arg} ${arg_names})
    if (is_arg_name)
      set(${prefix}_${current_arg_name} ${current_arg_list})
      set(current_arg_name ${arg})
      set(current_arg_list)
    else (is_arg_name)
      list_contains(is_option ${arg} ${option_names})
      if (is_option)
        set(${prefix}_${arg} TRUE)
      else (is_option)
        set(current_arg_list ${current_arg_list} ${arg})
      endif (is_option)
    endif (is_arg_name)
  endforeach(arg)
  set(${prefix}_${current_arg_name} ${current_arg_list})
endmacro(parse_arguments)

function (colormsg)
  string (ASCII 27 _escape)
  set(WHITE "29")
  set(GRAY "30")
  set(RED "31")
  set(GREEN "32")
  set(YELLOW "33")
  set(BLUE "34")
  set(MAG "35")
  set(CYAN "36")

  foreach (color WHITE GRAY RED GREEN YELLOW BLUE MAG CYAN)
    set(HI${color} "1\;${${color}}")
    set(LO${color} "2\;${${color}}")
    set(_${color}_ "4\;${${color}}")
    set(_HI${color}_ "1\;4\;${${color}}")
    set(_LO${color}_ "2\;4\;${${color}}")
  endforeach()

  set(str "")
  set(coloron FALSE)
  foreach(arg ${ARGV})
    if (NOT ${${arg}} STREQUAL "")
      if (CMAKE_COLOR_MAKEFILE)
        set(str "${str}${_escape}[${${arg}}m")
        set(coloron TRUE)
      endif()
    else()
      set(str "${str}${arg}")
      if (coloron)
        set(str "${str}${_escape}[0m")
        set(coloron FALSE)
      endif()
      set(str "${str} ")
    endif()
  endforeach()
  message(STATUS ${str})
endfunction()
