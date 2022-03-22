set(jvm_launcher_SOURCES jvm-launcher.c)

add_custom_target(native-testers DEPENDS jvm-launcher)
add_dependencies(build-all native-testers)

add_executable(jvm-launcher ${jvm_launcher_SOURCES})

string(REPLACE "." "/" STARPUJNI_LAUNCHER_CLASS "${STARPUJNI_LAUNCHER_CLASS}")

target_compile_definitions(jvm-launcher PUBLIC
        "STARPUJNI_JARFILE=\"${STARPUJNI_JARFILE}\""
        "STARPUJNI_LAUNCHER_CLASS=\"${STARPUJNI_LAUNCHER_CLASS}\"")

include_directories(jvm-launcher "${JNI_INCLUDE_DIRS}")
target_link_libraries(jvm-launcher "${JNI_LIBRARIES}")
