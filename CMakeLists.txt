cmake_minimum_required(VERSION 2.8)
project(MethodBackPropagation)

file(GLOB CPPS "src/*.cpp")
file(GLOB HS "include/*.h")
include_directories(include)

add_executable(${PROJECT_NAME} ${CPPS} ${HS})

set_property(GLOBAL PROPERTY USE_FOLDERS ON)