cmake_minimum_required(VERSION 3.5)
SET(CMAKE_CXX_FLAGS "-std=c++11 -O3")
set(LLVM_LINK_COMPONENTS support)
project(cfg)

find_package(Clang REQUIRED)

add_executable(cfg
  ac_parser.cpp
  )
target_link_libraries(cfg
  PRIVATE
  clangTooling
  clangBasic
  clangASTMatchers
  )
link_directories(
  "/usr/lib/llvm-9/lib/"
 
)

include_directories(
  "/usr/lib/llvm-9/include/"
  )