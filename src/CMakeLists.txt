#Copyright(c) Microsoft Corporation.All rights reserved.
#Licensed under the MIT                        license.

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_COMPILE_WARNING_AS_ERROR ON)

if(MSVC)
    add_subdirectory(dll)
else()
    #file(GLOB CPP_SOURCES *.cpp)
    set(CPP_SOURCES abstract_data_store.cpp ann_exception.cpp disk_utils.cpp 
        distance.cpp index.cpp in_mem_graph_store.cpp in_mem_data_store.cpp
        linux_aligned_file_reader.cpp math_utils.cpp natural_number_map.cpp
        in_mem_data_store.cpp in_mem_graph_store.cpp
        natural_number_set.cpp memory_mapper.cpp partition.cpp pq.cpp
        pq_flash_index.cpp scratch.cpp logger.cpp utils.cpp filter_utils.cpp index_factory.cpp abstract_index.cpp pq_l2_distance.cpp pq_data_store.cpp)
    if (RESTAPI)
        list(APPEND CPP_SOURCES restapi/search_wrapper.cpp restapi/server.cpp)
    endif()
    add_library(${PROJECT_NAME} ${CPP_SOURCES})
    add_library(${PROJECT_NAME}_s STATIC ${CPP_SOURCES})

    # 1. NUMA 라이브러리 찾기
    find_library(NUMA_LIB NAMES numa)

    if(NOT NUMA_LIB)
        message(FATAL_ERROR "libnuma not found. Please install it (e.g., sudo apt install libnuma-dev)")
    endif()

    # 2. 타겟에 라이브러리 링크
    # ${PROJECT_NAME} (공유 라이브러리)와 ${PROJECT_NAME}_s (정적 라이브러리) 모두에 링크합니다.
    target_link_libraries(${PROJECT_NAME} PRIVATE ${NUMA_LIB})
    target_link_libraries(${PROJECT_NAME}_s PRIVATE ${NUMA_LIB})
endif()

if (NOT MSVC)
    install(TARGETS ${PROJECT_NAME} LIBRARY)
endif()
