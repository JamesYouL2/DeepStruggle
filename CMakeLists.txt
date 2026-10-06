cmake_minimum_required(VERSION 3.20)

# The engine is built with clang. Measured on the same sources and flags, clang 21 against
# GCC 15: +20-24% on raw stepping and on the batch runner's step + observation path, with the two
# builds playing bit-identical games. So clang is chosen here unless a compiler is named
# explicitly, and anything else is refused below rather than built slower without a word.
# ~/.local/bin is searched for the wrappers tools/scripts/install_clang_userspace.sh writes on a
# machine without root.
# A toolchain file (emcmake's, for the browser build) names its own compiler.
if(NOT DEFINED CMAKE_CXX_COMPILER AND NOT DEFINED ENV{CXX} AND NOT CMAKE_TOOLCHAIN_FILE)
    find_program(TS_CLANGXX
        NAMES clang++ clang++-22 clang++-21 clang++-20 clang++-19 clang++-18
        HINTS "$ENV{HOME}/.local/bin")
    if(TS_CLANGXX)
        set(CMAKE_CXX_COMPILER "${TS_CLANGXX}")
    endif()
endif()

project(ts_ai CXX)

if(NOT CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    message(FATAL_ERROR
        "ts_ai builds the engine with clang, but this build directory uses "
        "${CMAKE_CXX_COMPILER_ID} (${CMAKE_CXX_COMPILER}).\n"
        "  - install clang: apt-get install clang   (no root: tools/scripts/install_clang_userspace.sh)\n"
        "  - then configure afresh -- the compiler is fixed when a build directory is created:\n"
        "      cmake --fresh -B build/release -S . -DPython_EXECUTABLE=$(pwd)/.venv/bin/python3\n"
        "    (tools/scripts/check_engine_fresh.sh does this by itself for an existing build dir)")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Optimization Flags
if(MSVC)
    add_compile_options(/O2 /arch:AVX2 /we4834)  # 4834 = discarding a [[nodiscard]] value
elseif(EMSCRIPTEN)
    # The browser build (tools/scripts/build_web.sh): WebAssembly has no -march to tune for.
    add_compile_options(-O3 -Wall -Wextra -Wpedantic -Wno-unused-parameter -Werror=unused-result)
else()
    # -march=native also tunes for this CPU: with -march and no -mtune clang tunes for the target
    # CPU, and -mtune=native was checked to produce byte-identical objects.
    add_compile_options(-O3 -march=native -Wall -Wextra -Wpedantic -Wno-unused-parameter
        # A discarded step() verdict is how a refused action becomes a silent no-op or an
        # unbounded retry loop. [[nodiscard]] only warns unless this promotes it.
        -Werror=unused-result)

    # Link-time optimisation. The engine's hot paths call across source files -- the observation
    # into the rules (ops.cpp, scoring.cpp), everything into MapData::get_country -- and without
    # it none of that can be inlined. Same sources, same games (the observation golden test and
    # tests/web/test_wasm_engine.py, which plays the native engine against the unoptimised wasm
    # one, hold it to that).
    option(TS_LTO "Build with link-time optimisation" ON)
    # thin and full measured the same on a single-threaded search (C++ tree 3.50 s against 3.49 s,
    # 3.82 s without LTO); thin links in parallel.
    set(TS_LTO_MODE "thin" CACHE STRING "-flto=<mode>: thin or full")
    if(TS_LTO)
        add_compile_options(-flto=${TS_LTO_MODE})
        add_link_options(-flto=${TS_LTO_MODE})
        # A static library of LTO objects holds LLVM bitcode, which the archiver has to be able
        # to read to index it: llvm-ar can, and GNU ar can through LLVM's gold plugin -- the only
        # one tools/scripts/install_clang_userspace.sh brings along.
        get_filename_component(TS_CLANG_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        find_program(TS_LLVM_AR NAMES llvm-ar llvm-ar-22 llvm-ar-21 llvm-ar-20 HINTS "${TS_CLANG_DIR}")
        find_program(TS_LLVM_RANLIB NAMES llvm-ranlib llvm-ranlib-22 llvm-ranlib-21 llvm-ranlib-20
                     HINTS "${TS_CLANG_DIR}")
        if(TS_LLVM_AR AND TS_LLVM_RANLIB)
            set(CMAKE_AR "${TS_LLVM_AR}" CACHE FILEPATH "" FORCE)
            set(CMAKE_RANLIB "${TS_LLVM_RANLIB}" CACHE FILEPATH "" FORCE)
        else()
            execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=LLVMgold.so
                            OUTPUT_VARIABLE TS_LLVMGOLD OUTPUT_STRIP_TRAILING_WHITESPACE)
            if(NOT EXISTS "${TS_LLVMGOLD}")
                message(FATAL_ERROR
                    "TS_LTO needs llvm-ar or LLVM's gold plugin (LLVMgold.so) to archive LTO "
                    "objects, and found neither next to ${CMAKE_CXX_COMPILER}. Install llvm "
                    "(apt-get install llvm), or configure with -DTS_LTO=OFF.")
            endif()
            set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> --plugin ${TS_LLVMGOLD} qc <TARGET> <OBJECTS>")
            set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> --plugin ${TS_LLVMGOLD} q <TARGET> <OBJECTS>")
            set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> --plugin ${TS_LLVMGOLD} <TARGET>")
        endif()
    endif()
endif()

add_subdirectory(engine)
add_subdirectory(bindings)
