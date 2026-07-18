cmake_minimum_required(VERSION 3.20)
project(see-through-cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# diffusers UNet state-dict names exceed ggml's default 64-char cap
add_compile_definitions(GGML_MAX_NAME=128)

add_subdirectory(ggml)

add_library(seethrough STATIC
    src/ops.cpp
    src/vae.cpp
    src/clip.cpp
    src/unet_frame.cpp
    src/scheduler.cpp
    src/lama.cpp
    src/psd_writer.cpp
    src/image_utils.cpp
)
target_link_libraries(seethrough PUBLIC ggml)
target_include_directories(seethrough PUBLIC src stb)

foreach(t test_trans_vae test_sd_vae test_trans_vae_full test_clip test_vae_encode test_unet_blocks test_unet_forward test_scheduler test_layerdiff_e2e test_ddim test_marigold_e2e test_lama test_psd)
    add_executable(${t} src/${t}.cpp)
    target_link_libraries(${t} PRIVATE seethrough)
endforeach()
