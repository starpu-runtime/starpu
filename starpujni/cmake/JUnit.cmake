include(FindJava)

#   add_junit_test(<target name> 
#       CLASSPATH [path1 ...]
#       TESTS [class1 ...]
#   )
function(add_junit_test TARGET_NAME)

   if (WIN32 AND NOT CYGWIN)
      set(SEPARATOR ";")
   else (WIN32 AND NOT CYGWIN)
      set(SEPARATOR ":")
   endif(WIN32 AND NOT CYGWIN)

   foreach (ARG ${ARGN})
      if (ARG MATCHES "CLASSPATH" OR ARG MATCHES "TESTS" OR ARG MATCHES "JVMARGS")
         set(TYPE ${ARG})

      else ()

         if (TYPE MATCHES "CLASSPATH")
            set(CLASSPATH "${CLASSPATH}${SEPARATOR}${ARG}")

         elseif (TYPE MATCHES "TESTS")
            set(TESTS ${TESTS} ${ARG})

         elseif (TYPE MATCHES "JVMARGS")
            set(JVMARGS ${JVMARGS} ${ARG})

         endif()

      endif()

   endforeach(ARG)

   add_test(NAME ${TARGET_NAME} COMMAND ${Java_JAVA_EXECUTABLE} ${JVMARGS} -classpath ${CLASSPATH} org.junit.runner.JUnitCore ${TESTS})

endfunction(add_junit_test)
