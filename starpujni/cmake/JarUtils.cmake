find_package(Java REQUIRED)

# Usage :
#
#   update_jar(target [ALL]
#     ARCHIVE filename.jar
#     FILES file ...
#     [WORKING_DIRECTORY path]
#     [DEPENDS depend ...]
#   )
function(add_update_jar JAR_TARGET)
    unset(JAR_ALL)
    unset(JAR_ARCHIVE)
    unset(JAR_FILES)
    unset(JAR_DEPENDS)
    unset(JAR_WORKING_DIRECTORY)

    set(options ALL)
    set(oneValueArgs ARCHIVE WORKING_DIRECTORY)
    set(multiValueArgs FILES DEPENDS)

    cmake_parse_arguments(JAR "${options}" "${oneValueArgs}" "${multiValueArgs}"
            ${ARGN})

    if (NOT JAR_FILES)
        message(WARNING "${JAR_TARGET}: No file added to jar.")
        return()
    endif ()

    if (NOT JAR_ARCHIVE)
        message(FATAL_ERROR "${JAR_TARGET}: No archive name specified for update.")
    endif ()

    if (NOT TARGET ${JAR_TARGET})
        if (JAR_ALL)
            set(ALL "ALL")
        endif ()
        add_custom_target(${JAR_TARGET} ${ALL} DEPENDS ${JAR_DEPENDS})
    endif ()

    add_custom_command(
            TARGET ${JAR_TARGET} POST_BUILD
            DEPENDS ${JAR_DEPENDS}
            WORKING_DIRECTORY "${JAR_WORKING_DIRECTORY}"
            COMMAND "${Java_JAR_EXECUTABLE}" "uvf" "${JAR_ARCHIVE}" ${JAR_FILES}
    )
endfunction()
