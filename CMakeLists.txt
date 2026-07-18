cmake_minimum_required(VERSION 3.20)
project(see-through-cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_subdirectory(ggml)

add_executable(test_trans_vae src/test_trans_vae.cpp)
target_link_libraries(test_trans_vae PRIVATE ggml)

add_executable(test_sd_vae src/test_sd_vae.cpp)
target_link_libraries(test_sd_vae PRIVATE ggml)

add_executable(test_trans_vae_full src/test_trans_vae_full.cpp)
target_link_libraries(test_trans_vae_full PRIVATE ggml)
