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
        # LTO objects are LLVM bitcode, and two tools have to read them:
        #  * the LINKER: GNU ld only through LLVM's gold plugin (LLVMgold.so), or lld natively;
        #  * the ARCHIVER, for the static libraries: llvm-ar, or GNU ar through the same plugin.
        # `apt-get install clang` brings neither the plugin nor llvm-ar on Ubuntu (the GitHub
        # runner failed here); `apt-get install llvm lld` brings both. The userspace installer
        # (tools/scripts/install_clang_userspace.sh) brings the plugin. Debian names the tools
        # with the version (llvm-ar-18) and keeps them in /usr/lib/llvm-<N>/bin, next to the real
        # compiler /usr/bin/clang++ links to.
        string(REGEX MATCH "^[0-9]+" TS_CLANG_MAJOR "${CMAKE_CXX_COMPILER_VERSION}")
        get_filename_component(TS_CLANG_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
        get_filename_component(TS_CLANG_REAL "${CMAKE_CXX_COMPILER}" REALPATH)
        get_filename_component(TS_CLANG_REAL_DIR "${TS_CLANG_REAL}" DIRECTORY)
        set(TS_LLVM_HINTS "${TS_CLANG_DIR}" "${TS_CLANG_REAL_DIR}" "/usr/lib/llvm-${TS_CLANG_MAJOR}/bin")
        find_program(TS_LLVM_AR NAMES llvm-ar-${TS_CLANG_MAJOR} llvm-ar HINTS ${TS_LLVM_HINTS})
        find_program(TS_LLVM_RANLIB NAMES llvm-ranlib-${TS_CLANG_MAJOR} llvm-ranlib HINTS ${TS_LLVM_HINTS})
        find_program(TS_LLD NAMES ld.lld-${TS_CLANG_MAJOR} ld.lld HINTS ${TS_LLVM_HINTS})
        execute_process(COMMAND "${CMAKE_CXX_COMPILER}" -print-file-name=LLVMgold.so
                        OUTPUT_VARIABLE TS_LLVMGOLD OUTPUT_STRIP_TRAILING_WHITESPACE)
        if(NOT IS_ABSOLUTE "${TS_LLVMGOLD}" OR NOT EXISTS "${TS_LLVMGOLD}")
            set(TS_LLVMGOLD "")
        endif()
        string(CONCAT TS_LTO_HELP
            "Install LLVM's tools next to clang -- on Debian/Ubuntu `apt-get install llvm lld`, "
            "without root tools/scripts/install_clang_userspace.sh -- or configure with -DTS_LTO=OFF.")

        # Linker: GNU ld with the plugin when it is there, lld otherwise.
        set(TS_LTO_LINK_OPTIONS "")
        set(TS_LTO_LINKER "the system linker with ${TS_LLVMGOLD}")
        if(NOT TS_LLVMGOLD)
            if(NOT TS_LLD)
                message(FATAL_ERROR "TS_LTO: the linker cannot read LTO objects -- neither LLVM's "
                    "gold plugin (LLVMgold.so) nor ld.lld was found for ${CMAKE_CXX_COMPILER} "
                    "(clang ${CMAKE_CXX_COMPILER_VERSION}). ${TS_LTO_HELP}")
            endif()
            set(TS_LTO_LINK_OPTIONS "--ld-path=${TS_LLD}")
            set(TS_LTO_LINKER "${TS_LLD}")
            add_link_options(${TS_LTO_LINK_OPTIONS})
        endif()

        # Archiver: llvm-ar when it is there, GNU ar through the plugin otherwise.
        if(TS_LLVM_AR AND TS_LLVM_RANLIB)
            set(CMAKE_AR "${TS_LLVM_AR}" CACHE FILEPATH "" FORCE)
            set(CMAKE_RANLIB "${TS_LLVM_RANLIB}" CACHE FILEPATH "" FORCE)
            set(TS_LTO_ARCHIVER "${TS_LLVM_AR}")
        elseif(TS_LLVMGOLD)
            set(TS_LTO_ARCHIVER "${CMAKE_AR} with ${TS_LLVMGOLD}")
            set(CMAKE_CXX_ARCHIVE_CREATE "<CMAKE_AR> --plugin ${TS_LLVMGOLD} qc <TARGET> <OBJECTS>")
            set(CMAKE_CXX_ARCHIVE_APPEND "<CMAKE_AR> --plugin ${TS_LLVMGOLD} q <TARGET> <OBJECTS>")
            set(CMAKE_CXX_ARCHIVE_FINISH "<CMAKE_RANLIB> --plugin ${TS_LLVMGOLD} <TARGET>")
        else()
            message(FATAL_ERROR "TS_LTO: the static libraries cannot be archived -- neither "
                "llvm-ar nor LLVM's gold plugin (LLVMgold.so) was found for ${CMAKE_CXX_COMPILER} "
                "(clang ${CMAKE_CXX_COMPILER_VERSION}). ${TS_LTO_HELP}")
        endif()

        # And prove the linker half here, so a toolchain that cannot link LTO objects stops at
        # configure with this message rather than deep in the build with a plugin error.
        include(CheckCXXSourceCompiles)
        set(CMAKE_REQUIRED_FLAGS "-flto=${TS_LTO_MODE}")
        set(CMAKE_REQUIRED_LINK_OPTIONS "-flto=${TS_LTO_MODE};${TS_LTO_LINK_OPTIONS}")
        check_cxx_source_compiles("int f(int x) { return x + 1; } int main() { return f(-1); }"
                                  TS_LTO_LINKS)
        unset(CMAKE_REQUIRED_FLAGS)
        unset(CMAKE_REQUIRED_LINK_OPTIONS)
        if(NOT TS_LTO_LINKS)
            message(FATAL_ERROR "TS_LTO: clang could not link a program with -flto=${TS_LTO_MODE} "
                "(see CMakeFiles/CMakeConfigureLog.yaml). ${TS_LTO_HELP}")
        endif()
        message(STATUS "LTO: -flto=${TS_LTO_MODE}; linking with ${TS_LTO_LINKER}; "
            "archiving with ${TS_LTO_ARCHIVER}")
    endif()
endif()

add_subdirectory(engine)
add_subdirectory(bindings)
