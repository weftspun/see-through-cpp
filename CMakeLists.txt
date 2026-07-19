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

# separate from seethrough_c above: that one is scoped to Lean witness
# testing (st_witness_check*), this one drives the real production
# pipeline (image bytes in, SVG + PNG layers out) for embedding in a
# server or other host process
add_library(see_through_c SHARED src/see_through_capi.cpp)
target_link_libraries(see_through_c PRIVATE seethrough)
target_compile_definitions(see_through_c PRIVATE ST2_BUILD_DLL)

add_executable(see-through-server src/server.cpp)
target_link_libraries(see-through-server PRIVATE see_through_c)
target_include_directories(see-through-server PRIVATE src third_party/cpp-httplib)
if (WIN32)
    target_link_libraries(see-through-server PRIVATE ws2_32 crypt32)
endif()


foreach(t test_trans_vae test_sd_vae test_trans_vae_full test_clip test_vae_encode test_vae_encode_160 test_unet_blocks test_unet_forward test_unet_forward_160 test_scheduler test_layerdiff_e2e test_layerdiff_e2e_160 test_ddim test_marigold_e2e test_lama graph_mem)
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

add_executable(probe_mul_mat_kahan tests/probe_mul_mat_kahan.cpp)
target_link_libraries(probe_mul_mat_kahan PRIVATE seethrough)

add_executable(psd_layers tests/psd_layers.cpp)
target_link_libraries(psd_layers PRIVATE Psd)
target_include_directories(psd_layers PRIVATE src stb)
target_sources(psd_layers PRIVATE src/image_utils.cpp)

# --- picoquic + picotls + mbedtls: vendored from fire/webtransportd
# (thirdparty/{picoquic,picotls,mbedtls}, commit b53faa2), same recipe as
# weft-warp-loop's flow-toolchain/CMakeLists.txt -- EXCEPT sockloop.c/
# winsockloop.c are INCLUDED here (not excluded): this server owns its own
# sockets via picoquic's own packet loop, there's no Flow event loop to
# defer to.
option(SEETHROUGH_BUILD_H3_SERVER "Build the HTTP/3 (picoquic) server" ON)
if (SEETHROUGH_BUILD_H3_SERVER)
    set(PICOQUIC_DEFS
        PICOQUIC_WITH_MBEDTLS=1
        PTLS_WITHOUT_OPENSSL=1
        PTLS_WITHOUT_FUSION=1
        PICOQUIC_WITHOUT_SSLKEYLOG=1
        DISABLE_DEBUG_PRINTF=1
    )
    if (WIN32)
        list(APPEND PICOQUIC_DEFS _WINDOWS)
    endif()

    set(VENDOR_QUIC_INCLUDES
        third_party/picoquic/picoquic
        third_party/picoquic/picohttp
        third_party/picoquic/picoquic_mbedtls
        third_party/picoquic/loglib
        third_party/picotls/include
        third_party/picotls/deps/cifra/src
        third_party/picotls/deps/cifra/src/ext
        third_party/picotls/deps/micro-ecc
        third_party/mbedtls/include
    )
    if (WIN32)
        list(INSERT VENDOR_QUIC_INCLUDES 0 third_party/picotls/picotlsvs/picotls)
    endif()

    file(GLOB PICOQUIC_CORE_SRCS "third_party/picoquic/picoquic/*.c")
    file(GLOB PICOHTTP_SRCS "third_party/picoquic/picohttp/*.c")

    file(GLOB PICOTLS_SRCS "third_party/picotls/lib/*.c")
    foreach(_ex certificate_compression.c fusion.c mbedtls.c mbedtls_sign.c openssl.c)
        list(REMOVE_ITEM PICOTLS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/picotls/lib/${_ex}")
    endforeach()
    if (WIN32)
        list(APPEND PICOTLS_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/picotls/picotlsvs/picotls/wintimeofday.c")
    endif()

    file(GLOB PICOTLS_CIFRA_SRCS "third_party/picotls/lib/cifra/*.c")
    list(REMOVE_ITEM PICOTLS_CIFRA_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/picotls/lib/cifra/libaegis.c")

    file(GLOB CIFRA_SRCS "third_party/picotls/deps/cifra/src/*.c")
    file(GLOB CIFRA_TEST_SRCS "third_party/picotls/deps/cifra/src/test*.c")
    list(REMOVE_ITEM CIFRA_SRCS ${CIFRA_TEST_SRCS})
    foreach(_ex donna naclref tweetnacl)
        list(REMOVE_ITEM CIFRA_SRCS
            "${CMAKE_CURRENT_SOURCE_DIR}/third_party/picotls/deps/cifra/src/curve25519.${_ex}.c")
    endforeach()

    file(GLOB MICROECC_SRCS "third_party/picotls/deps/micro-ecc/*.c")
    file(GLOB PICOQUIC_MBEDTLS_SRCS "third_party/picoquic/picoquic_mbedtls/*.c")

    file(GLOB MBEDTLS_SRCS "third_party/mbedtls/library/*.c")
    list(REMOVE_ITEM MBEDTLS_SRCS
        "${CMAKE_CURRENT_SOURCE_DIR}/third_party/mbedtls/library/godot_core_mbedtls_platform.c")

    file(GLOB LOGLIB_SRCS "third_party/picoquic/loglib/*.c")

    add_library(picoquic_vendored STATIC
        ${PICOQUIC_CORE_SRCS} ${PICOHTTP_SRCS}
        ${PICOTLS_SRCS} ${PICOTLS_CIFRA_SRCS} ${CIFRA_SRCS} ${MICROECC_SRCS}
        ${PICOQUIC_MBEDTLS_SRCS} ${MBEDTLS_SRCS} ${LOGLIB_SRCS})
    target_compile_options(picoquic_vendored PRIVATE -w)
    if (WIN32 AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
        target_compile_options(picoquic_vendored PRIVATE
            -Wno-error=incompatible-pointer-types
            -Wno-error=implicit-function-declaration
            -Wno-error=int-conversion)
    endif()
    target_compile_definitions(picoquic_vendored PRIVATE ${PICOQUIC_DEFS})
    target_include_directories(picoquic_vendored SYSTEM PUBLIC ${VENDOR_QUIC_INCLUDES})
    find_package(Threads REQUIRED)
    target_link_libraries(picoquic_vendored PUBLIC Threads::Threads)
    if (WIN32)
        target_link_libraries(picoquic_vendored PUBLIC ws2_32 iphlpapi bcrypt)
    endif()

    add_executable(see-through-server-h3 src/server_h3.cpp)
    target_compile_definitions(see-through-server-h3 PRIVATE ${PICOQUIC_DEFS})
    target_include_directories(see-through-server-h3 PRIVATE src)
    if (WIN32 AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
        target_compile_options(see-through-server-h3 PRIVATE
            -Wno-error=incompatible-pointer-types
            -Wno-error=implicit-function-declaration
            -Wno-error=int-conversion)
    endif()
    target_link_libraries(see-through-server-h3 PRIVATE see_through_c picoquic_vendored)

    # dev-only test client (vendored picoquic reference CLI, client mode) --
    # not part of the product, just for exercising see-through-server-h3
    # without a real browser/H3 client on hand
    add_executable(picoquicdemo_client
        third_party/picoquic/picoquicfirst/picoquicdemo.c
        third_party/picoquic/picoquicfirst/getopt.c
        third_party/picoquic/picohttp/demoserver.c)
    target_compile_definitions(picoquicdemo_client PRIVATE ${PICOQUIC_DEFS})
    target_include_directories(picoquicdemo_client PRIVATE ${VENDOR_QUIC_INCLUDES}
        third_party/picoquic/picoquicfirst)
    if (WIN32 AND NOT (CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
        target_compile_options(picoquicdemo_client PRIVATE
            "SHELL:-include stdlib.h" "SHELL:-include malloc.h"
            "SHELL:-include Winsock2.h" "SHELL:-include ws2tcpip.h"
            -Wno-error=incompatible-pointer-types
            -Wno-error=implicit-function-declaration
            -Wno-error=int-conversion)
    endif()
    target_link_libraries(picoquicdemo_client PRIVATE picoquic_vendored)
endif()
