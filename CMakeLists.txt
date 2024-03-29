cmake_minimum_required(VERSION 3.16)
set(CMAKE_CXX_STANDARD 20)

project(XeTLA)

include(CTest)
enable_testing()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/tools/cmake")
find_package(MKL REQUIRED)
find_package(GTest REQUIRED)

# For large registers mode, enable 256 registers for kernels
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -vc-disable-indvars-opt")
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -vc-codegen")
# For registers usage verbose at AOT
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -printregusage")
# Enable bank conflict reduction.
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -enableBCR")
# Optimization to reduce the tokens used for DPAS instruction.
set(XETLA_OFFLINE_OPTIONS "${XETLA_OFFLINE_OPTIONS} -Xfinalizer -DPASTokenReduction")

set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} -Xs)
if (USE_AOT_DEVLIST)
    set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} "-device pvc -options '${XETLA_OFFLINE_OPTIONS}'")
else()
    set(XETLA_KERNEL_FLAGS ${XETLA_KERNEL_FLAGS} "${XETLA_OFFLINE_OPTIONS}")
endif()

# debug option
message(STATUS "'DEBUG' is set to " ${DEBUG})
if (${DEBUG})
    #add_compile_options(-debug=minimal -Rno-debug-disables-optimization -DDEBUG=${DEBUG})
    add_compile_options(-debug=minimal -Rno-debug-disables-optimization -DDEBUG)
endif ()

# log message print
set(LOG "off" CACHE STRING "on/off log message")
message(STATUS "log message is " ${LOG})
if (${LOG} STREQUAL "on")
    add_definitions(-DLOG_PRINT)
endif ()

include_directories(${GTEST_INCLUDE_DIR})
#add_compile_options(-fsycl -fp-model=precise -Wall -Wextra -Werror)
add_compile_options(-fsycl -fp-model=precise -Wno-c++11-narrowing -O3)
add_link_options(-fsycl -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm)
link_libraries(GTest::gtest_main)
include_directories($ENV{DNNLROOT}/include)
link_directories($ENV{DNNLROOT}/lib)
link_libraries(-ldnnl)
#link_libraries(-lgtest -lgtest_main)

add_subdirectory(tests)
add_subdirectory(examples)
