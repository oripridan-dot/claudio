cmake_minimum_required(VERSION 3.18)
project(ClaudioIntentExtractor VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(intent_core SHARED intent_extractor.cpp)

if(EMSCRIPTEN)
    # Output to .js (which generates both intent_core.wasm and intent_core.js)
    set_target_properties(intent_core PROPERTIES SUFFIX ".js")
    
    # Export functions and enable memory growth
    target_link_options(intent_core PRIVATE
        "-s" "EXPORTED_RUNTIME_METHODS=['cwrap','ccall']"
        "-s" "EXPORTED_FUNCTIONS=['_computeRMS','_autocorrelationF0','_computeSpectralCentroid','_initFilterbank','_computeMFCC','_malloc','_free']"
        "-s" "ALLOW_MEMORY_GROWTH=1"
        "-s" "MODULARIZE=1"
        "-s" "EXPORT_NAME='createIntentModule'"
    )
endif()
