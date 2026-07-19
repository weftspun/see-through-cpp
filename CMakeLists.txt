cmake_minimum_required(VERSION 3.20)
project(see-through-cpp CXX C)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# diffusers UNet state-dict names exceed ggml's default 64-char cap
add_compile_definitions(GGML_MAX_NAME=128)

add_subdirectory(third_party/ggml)

# psd_sdk's own CMakeLists.txt requires a cmake_minimum_required too old for
# current CMake; compile its sources directly instead of add_subdirectory
# (read-only usage: PsdExport*/PsdNativeFile writer bits aren't needed for
# our upstream-parity reader, but bringing the whole psd_source list in is
# simplest and matches how the upstream project itself builds it)
set(PSD_SDK_DIR third_party/psd_sdk/src/Psd)
file(GLOB PSD_SDK_SOURCES ${PSD_SDK_DIR}/*.cpp ${PSD_SDK_DIR}/*.c)
if (WIN32)
    list(FILTER PSD_SDK_SOURCES EXCLUDE REGEX ".*NativeFile_(Mac|Linux)\\.cpp$")
else()
    list(FILTER PSD_SDK_SOURCES EXCLUDE REGEX ".*NativeFile\\.cpp$")
endif()
add_library(Psd STATIC ${PSD_SDK_SOURCES})
target_include_directories(Psd PUBLIC third_party/psd_sdk/src)

add_library(seethrough STATIC
    src/ops.cpp
    src/vae.cpp
    src/clip.cpp
    src/unet_frame.cpp
    src/scheduler.cpp
    src/lama.cpp
    src/image_utils.cpp
    src/postproc.cpp
    src/pipeline.cpp
)
target_link_libraries(seethrough PUBLIC ggml)
target_include_directories(seethrough PUBLIC src stb)

add_executable(see-through src/see_through.cpp)
target_link_libraries(see-through PRIVATE seethrough)

add_library(seethrough_c SHARED src/seethrough_capi.cpp)
target_link_libraries(seethrough_c PRIVATE seethrough)
target_compile_definitions(seethrough_c PRIVATE ST_BUILD_DLL)
# ggml's own Q4_0 reference quantizer, for the quantization witness cases
# (private header: block_q4_0 layout + quantize_row_q4_0_ref)
target_include_directories(seethrough_c PRIVATE third_party/ggml/src)


foreach(t test_trans_vae test_sd_vae test_trans_vae_full test_clip test_vae_encode test_unet_blocks test_unet_forward test_scheduler test_layerdiff_e2e test_ddim test_marigold_e2e test_lama graph_mem)
    add_executable(${t} tests/${t}.cpp)
    target_link_libraries(${t} PRIVATE seethrough)
endforeach()

add_executable(probe_flash_bigT tests/probe_flash_bigT.cpp)
target_link_libraries(probe_flash_bigT PRIVATE seethrough_c)
target_include_directories(probe_flash_bigT PRIVATE src)

add_executable(probe_attn_block_tiled tests/probe_attn_block_tiled.cpp)
target_link_libraries(probe_attn_block_tiled PRIVATE seethrough)

add_executable(probe_weight_stats tests/probe_weight_stats.cpp)
target_link_libraries(probe_weight_stats PRIVATE seethrough)

add_executable(psd_layers tests/psd_layers.cpp)
target_link_libraries(psd_layers PRIVATE Psd)
target_include_directories(psd_layers PRIVATE src stb)
target_sources(psd_layers PRIVATE src/image_utils.cpp)
