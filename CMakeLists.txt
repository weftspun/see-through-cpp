cmake_minimum_required(VERSION 3.20)
project(see-through-cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# diffusers UNet state-dict names exceed ggml's default 64-char cap
add_compile_definitions(GGML_MAX_NAME=128)

add_subdirectory(third_party/ggml)

add_library(seethrough STATIC
    src/ops.cpp
    src/vae.cpp
    src/clip.cpp
    src/unet_frame.cpp
    src/scheduler.cpp
    src/lama.cpp
    src/psd_writer.cpp
    src/image_utils.cpp
    src/postproc.cpp
    src/pipeline.cpp
)
target_link_libraries(seethrough PUBLIC ggml)
target_include_directories(seethrough PUBLIC src stb)

add_executable(see-through src/see_through.cpp)
target_link_libraries(see-through PRIVATE seethrough)

option(SEETHROUGH_TESTS "Build the validation and property tests" ON)
if (SEETHROUGH_TESTS)
    set(_st_shared ${BUILD_SHARED_LIBS})
    set(BUILD_SHARED_LIBS OFF)          # rapidcheck exports nothing as a DLL
    add_subdirectory(third_party/rapidcheck)
    set(BUILD_SHARED_LIBS ${_st_shared})
    add_executable(test_properties tests/test_properties.cpp)
    target_link_libraries(test_properties PRIVATE seethrough rapidcheck)
    add_executable(test_graph_properties tests/test_graph_properties.cpp)
    target_link_libraries(test_graph_properties PRIVATE seethrough rapidcheck)
endif()

foreach(t test_trans_vae test_sd_vae test_trans_vae_full test_clip test_vae_encode test_unet_blocks test_unet_forward test_scheduler test_layerdiff_e2e test_ddim test_marigold_e2e test_lama test_psd graph_mem)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE seethrough)
endforeach()
