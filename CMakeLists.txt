cmake_minimum_required(VERSION 3.15)

project(testcomms LANGUAGES Fortran)

option(USE_UVM "Use Managed memory" OFF)

set(CMAKE_Fortran_FLAGS "-cpp -D_USE_NVTX -lnvToolsExt")
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0")
if (USE_UVM)
    set(CMAKE_Fortran_FLAGS_RELEASE "-fast -gpu=cc61,managed,lineinfo -cuda -acc -Minfo=all")
else()
    set(CMAKE_Fortran_FLAGS_RELEASE "-fast -gpu=cc61,lineinfo -cuda -acc -Minfo=all")
endif()

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/src/*.f90)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)

add_executable(testcomms ${SRC_FILES})