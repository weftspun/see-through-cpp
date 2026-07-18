cmake_minimum_required(VERSION 3.20)
project(see-through-cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(ggml)

add_library(seethrough STATIC
    src/ops.cpp
    src/vae.cpp
    src/clip.cpp
)
target_link_libraries(seethrough PUBLIC ggml)
target_include_directories(seethrough PUBLIC src)

foreach(t test_trans_vae test_sd_vae test_trans_vae_full test_clip test_vae_encode)
    add_executable(${t} src/${t}.cpp)
    target_link_libraries(${t} PRIVATE seethrough)
endforeach()
