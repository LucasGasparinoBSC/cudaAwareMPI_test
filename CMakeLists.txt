cmake_minimum_required(VERSION 3.15)

set(ENV{FC} "mpif90")

project(testcomms LANGUAGES Fortran)

option(USE_UVM "Use Managed memory" OFF)

if (USE_UVM)
    set(CMAKE_Fortran_FLAGS "-cpp -gpu=cc61,managed,lineinfo -acc -cuda -D_USE_NVTX -lnvToolsExt -Minfo=all")
else()
    set(CMAKE_Fortran_FLAGS "-cpp -gpu=cc61,lineinfo -acc -cuda -D_USE_NVTX -lnvToolsExt -Minfo=all")
endif()
set(CMAKE_Fortran_FLAGS_DEBUG "-g -O0")
set(CMAKE_Fortran_FLAGS_RELEASE "-fast")

file(GLOB_RECURSE SRC_FILES ${CMAKE_CURRENT_LIST_DIR}/src/*.f90)
set(SRC_DIR ${CMAKE_CURRENT_LIST_DIR}/src)

add_executable(testcomms ${SRC_FILES})
