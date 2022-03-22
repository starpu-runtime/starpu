set(MAVEN_DEPENDENCY_PLUGIN_VERSION "2.4")

include(Utilities)
include(FindJava)

IF(${CMAKE_SYSTEM_NAME} MATCHES "Windows")
find_program(MAVEN
  mvn.bat
  )
else()
find_program(MAVEN
  mvn
  )
endif()
if(MAVEN)
  message(STATUS "Found maven: ${MAVEN}")
endif()

# This function allow you to create a ZIP file
#
# Usage :
#
#   create_zip(name
#     ARCHIVE file.zip
#     FILES file ...
#     [WORKING_DIRECTORY path]
#     [DEPENDS depend ...]
#   )
function(create_zip ZIP_TARGET)
  unset(ZIP_ARCHIVE)
  unset(ZIP_FILES)
  unset(ZIP_DEPENDS)
  unset(ZIP_WORKING_DIRECTORY)

  if(Java_JAR_EXECUTABLE)
    parse_arguments(ZIP "ARCHIVE;FILES;DEPENDS;WORKING_DIRECTORY;" "" ${ARGN})

    if(NOT ZIP_FILES)
      message(FATAL_ERROR "No files specified for zip")
    endif()
    if(NOT ZIP_ARCHIVE)
      message(FATAL_ERROR "No archive name specified for zip")
    endif()

    if(NOT TARGET ${ZIP_TARGET})
      add_custom_target(${ZIP_TARGET}
        DEPENDS ${ZIP_DEPENDS}
        )
    endif()

    add_custom_command(
      TARGET ${ZIP_TARGET}
      DEPENDS ${ZIP_DEPENDS}
      WORKING_DIRECTORY "${ZIP_WORKING_DIRECTORY}"
      COMMAND "${Java_JAR_EXECUTABLE}" "cfM" "${ZIP_ARCHIVE}" ${ZIP_FILES}
      )
  else()
    message(WARNING "Please, install Java")
  endif()
endfunction()

# This fiction allow you to deploy a file in a Maven repository
#
# usage : 
#
#  maven_deploy_file(target
#    FILE name
#    GROUP_ID groupId
#    ARTIFACT_ID artifactId
#    VERSION version
#    PACKAGING zip|jar
#    [REPOSITORY_ID repositoryId]
#    [URL repositoryUrl]
#    [DEPENDS depends ...]
#  )
#
# If repositoryId is not given, this function will use MAVEN_DEFAULT_REPOSITORY_ID
# If repositoryUrl is not given, this function will use MAVEN_DEFAULT_URL
function(maven_deploy_file MAVEN_TARGET)
  unset(MAVEN_FILE)
  unset(MAVEN_GROUP_ID)
  unset(MAVEN_ARTIFACT_ID)
  unset(MAVEN_VERSION)
  unset(MAVEN_PACKAGING)
  unset(MAVEN_DEPENDS)

  if(MAVEN)
    parse_arguments(MAVEN "FILE;GROUP_ID;ARTIFACT_ID;VERSION;PACKAGING;REPOSITORY_ID;DEPENDS;URL;" "" ${ARGN})

    if(NOT MAVEN_FILE)
      message(FATAL_ERROR "File not specified for maven_deploy_file")
    endif()
    if(NOT MAVEN_GROUP_ID)
      message(FATAL_ERROR "Group ID not specified for maven_deploy_file")
    endif()
    if(NOT MAVEN_ARTIFACT_ID)
      message(FATAL_ERROR "Artifact ID not specified for maven_deploy_file")
    endif()
    if(NOT MAVEN_VERSION)
      message(FATAL_ERROR "Version not specified for maven_deploy_file")
    endif()
    if(NOT MAVEN_REPOSITORY_ID)
      if(NOT MAVEN_DEFAULT_REPOSITORY_ID)
        message(FATAL_ERROR "Repository ID not specified for maven_deploy_file")
      else()
        set(MAVEN_REPOSITORY_ID ${MAVEN_DEFAULT_REPOSITORY_ID})
      endif()
    endif()
    if(NOT MAVEN_URL)
      if(NOT MAVEN_DEFAULT_URL)
        message(FATAL_ERROR "Repository URL not specified for maven_deploy_file")
      else()
        set(MAVEN_URL ${MAVEN_DEFAULT_URL})
      endif()
    endif()

    if(NOT TARGET ${MAVEN_TARGET})
      add_custom_target(${MAVEN_TARGET}
        DEPENDS ${MAVEN_DEPENDS}
        )
    endif()

    if(NOT MAVEN_PACKAGING)
      set(MAVEN_PACKAGING jar)
    elseif(MAVEN_PACKAGING MATCHES "zip")
      get_filename_component(ARCHIVE_PATH ${MAVEN_FILE} PATH)
      get_filename_component(ARCHIVE_NAME ${MAVEN_FILE} NAME_WE)
      get_filename_component(ZIP_CONTENT ${MAVEN_FILE} NAME)
      set(ZIP_FILENAME "${ARCHIVE_PATH}/${ARCHIVE_NAME}-${MAVEN_VERSION}.zip")
      set(MAVEN_FILE ${ZIP_FILENAME})
      create_zip(${MAVEN_TARGET} 
        ARCHIVE "${MAVEN_FILE}"
        FILES "${ZIP_CONTENT}"
        WORKING_DIRECTORY "${ARCHIVE_PATH}"
        )
    endif()

    add_custom_command(
      TARGET ${MAVEN_TARGET}
      POST_BUILD
      COMMAND 
        "${MAVEN}" "deploy:deploy-file" 
        "-DgroupId=${MAVEN_GROUP_ID}" 
        "-DartifactId=${MAVEN_ARTIFACT_ID}" 
        "-Dversion=${MAVEN_VERSION}" 
        "-Dpackaging=${MAVEN_PACKAGING}" 
        "-Dfile=${MAVEN_FILE}"
        "-DuniqueVersion=true"
        "-DupdateReleaseInfo=true"
        "-DrepositoryId=${MAVEN_REPOSITORY_ID}"
        "-Durl=${MAVEN_URL}"
      )
  else()
    message(WARNING "Please, install Maven")
  endif()
endfunction()

# This method allow you to get a dependency from a Maven repository
#
# usage :
#   maven_get_dependency(
#     GROUP_ID net.java.dev.jna
#     ARTIFACT_ID jna
#     VERSION 3.4.0
#     [PACKAGING jar]
#     [URL http://nexus.vidal.net:8081/nexus/content/groups/vidal-releases]
#     [DESTINATION /path/to/jna-3.4.0.jar]
#     [DEPENDS depends ...]
#     [TARGET target]
#     )
#
# If repositoryId is not given, this function will use MAVEN_DEFAULT_REPOSITORY_ID
# If repositoryUrl is not given, this function will use MAVEN_DEFAULT_URL
function(maven_get_dependency)
  unset(MAVEN_TARGET)
  unset(MAVEN_GROUP_ID)
  unset(MAVEN_ARTIFACT_ID)
  unset(MAVEN_VERSION)
  unset(MAVEN_PACKAGING)
  unset(MAVEN_DEPENDS)

  if(MAVEN)
    parse_arguments(MAVEN "TARGET;GROUP_ID;ARTIFACT_ID;VERSION;PACKAGING;DESTINATION;DEPENDS;URL;" "" ${ARGN})

    if(NOT MAVEN_GROUP_ID)
      message(FATAL_ERROR "Group ID not specified for maven_get_dependency")
    endif()
    if(NOT MAVEN_ARTIFACT_ID)
      message(FATAL_ERROR "Artifact ID not specified for maven_get_dependency")
    endif()
    if(NOT MAVEN_VERSION)
      message(FATAL_ERROR "Version not specified for maven_get_dependency")
    endif()
    if(NOT MAVEN_PACKAGING)
      set(MAVEN_PACKAGING "jar")
    endif()
    if(NOT MAVEN_REPOSITORY_ID)
      if(NOT MAVEN_DEFAULT_REPOSITORY_ID)
        message(FATAL_ERROR "Repository ID not specified for maven_get_dependency")
      else()
        set(MAVEN_REPOSITORY_ID ${MAVEN_DEFAULT_REPOSITORY_ID})
      endif()
    endif()
    if(NOT MAVEN_URL)
      if(NOT MAVEN_DEFAULT_URL)
        message(FATAL_ERROR "Repository URL not specified for maven_get_dependency")
      else()
        set(MAVEN_URL ${MAVEN_DEFAULT_URL})
      endif()
    endif()
    if(MAVEN_DESTINATION)
      set(MAVEN_DESTINATION "-Ddest=${MAVEN_DESTINATION}")
    endif()

    if(MAVEN_TARGET)
      if(NOT TARGET ${MAVEN_TARGET})
        add_custom_target(${MAVEN_TARGET}
          DEPENDS ${MAVEN_DEPENDS}
          )
      endif()
    endif()
    
    if(CMAKE_VERBOSE_MAKEFILE)
      set(MAVEN_VERBOSE)
    else()
      set(MAVEN_VERBOSE OUTPUT_QUIET)
    endif()

    if(MAVEN_TARGET)
      add_custom_command(
        TARGET ${MAVEN_TARGET} 
        PRE_BUILD
        COMMAND
          "${MAVEN}" "org.apache.maven.plugins:maven-dependency-plugin:${MAVEN_DEPENDENCY_PLUGIN_VERSION}:get"
          "-DgroupId=${MAVEN_GROUP_ID}"
          "-DartifactId=${MAVEN_ARTIFACT_ID}"
          "-Dversion=${MAVEN_VERSION}"
          "-Dpackaging=${MAVEN_PACKAGING}"
          "-DremoteRepositories=${MAVEN_URL}"
          ${MAVEN_DESTINATION}
        )
    else()
      message(STATUS "Get maven dependency ${MAVEN_GROUP_ID}:${MAVEN_ARTIFACT_ID}:${MAVEN_VERSION}:${MAVEN_PACKAGING}")
      execute_process(
        COMMAND 
          "${MAVEN}" "org.apache.maven.plugins:maven-dependency-plugin:${MAVEN_DEPENDENCY_PLUGIN_VERSION}:get"
          "-DgroupId=${MAVEN_GROUP_ID}"
          "-DartifactId=${MAVEN_ARTIFACT_ID}"
          "-Dversion=${MAVEN_VERSION}"
          "-Dpackaging=${MAVEN_PACKAGING}"
          "-DremoteRepositories=${MAVEN_URL}"
          ${MAVEN_DESTINATION}
          ${MAVEN_VERBOSE}
        )
    endif()
  endif()

endfunction()

