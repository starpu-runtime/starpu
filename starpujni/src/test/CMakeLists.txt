include(Maven)
include(JUnit)

#get_target_property(STARPUJNI_JARFILE starpujni-jar JAR_FILE)
#add_test(NAME TestJNIFoo
#        COMMAND ${MAVEN} test
#        WORKING_DIRECTORY ${PROJECT_SOURCE_DIR})

add_subdirectory(native)
