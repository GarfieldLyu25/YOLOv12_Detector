cmake_minimum_required(VERSION 3.15)
project(YOLOv12_Detector)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(MSVC)
    add_compile_options("$<$<C_COMPILER_ID:MSVC>:/utf-8>")
    add_compile_options("$<$<CXX_COMPILER_ID:MSVC>:/utf-8>")
    add_definitions(-DUNICODE -D_UNICODE)
endif()
# ==================== 路径设置 ====================
set(OPENCV_PATH "D:/Garfield_Lyu/postgraduate/project/YOLOv12-ONNX-TensorRT/deps/opencv")
set(ONNXRUNTIME_PATH "D:/Garfield_Lyu/postgraduate/project/YOLOv12-ONNX-TensorRT/deps/onnxruntime")

# ==================== OpenCV 配置 ====================
# 设置 OpenCV 查找路径
set(OpenCV_DIR "${OPENCV_PATH}/build")
find_package(OpenCV 4.12 REQUIRED)

if(NOT OpenCV_FOUND)
    message(FATAL_ERROR "❌ 未找到 OpenCV 4.12，请检查路径: ${OpenCV_DIR}")
else()
    message(STATUS "✓ 找到 OpenCV ${OpenCV_VERSION}")
    message(STATUS "OpenCV 包含目录: ${OpenCV_INCLUDE_DIRS}")
    message(STATUS "OpenCV 库目录: ${OpenCV_LIB_DIR}")
    message(STATUS "OpenCV 库文件: ${OpenCV_LIBS}")
endif()

# ==================== ONNX Runtime 配置 ====================
set(ONNXRUNTIME_INCLUDE_DIR "${ONNXRUNTIME_PATH}/include")
set(ONNXRUNTIME_LIB_DIR "${ONNXRUNTIME_PATH}/lib")

if(NOT EXISTS "${ONNXRUNTIME_INCLUDE_DIR}")
    message(FATAL_ERROR "❌ ONNX Runtime Include 目录不存在: ${ONNXRUNTIME_INCLUDE_DIR}")
endif()

if(NOT EXISTS "${ONNXRUNTIME_LIB_DIR}")
    message(FATAL_ERROR "❌ ONNX Runtime Lib 目录不存在: ${ONNXRUNTIME_LIB_DIR}")
endif()

message(STATUS "ONNX Runtime 包含目录: ${ONNXRUNTIME_INCLUDE_DIR}")
message(STATUS "ONNX Runtime 库目录: ${ONNXRUNTIME_LIB_DIR}")



# ==================== 链接目录 ====================
link_directories(
        ${OpenCV_LIB_DIR}
        ${ONNXRUNTIME_LIB_DIR}
)

# ==================== 源文件 ====================
file(GLOB SOURCES
        "src/main.cpp"
        "src/yolo_detector.cpp"
        "src/utils.cpp"
)

# ==================== 创建可执行文件 ====================
add_executable(yolo_detector ${SOURCES})

# ==================== 设置包含目录 ====================
target_include_directories(yolo_detector PRIVATE
        "${CMAKE_SOURCE_DIR}/include"
        ${ONNXRUNTIME_INCLUDE_DIR}
        ${OpenCV_INCLUDE_DIRS}
)
# ==================== 链接库 ====================
# ==================== 链接库 ====================
target_link_libraries(yolo_detector
        ${OpenCV_LIBS}
        onnxruntime. lib
)

message(STATUS "✓ 链接 OpenCV 和 ONNX Runtime 库")

# ==================== 编译选项 ====================
if(MSVC)
    # 设置字符集为 UTF-8
    target_compile_options(yolo_detector PRIVATE /utf-8)
    target_compile_options(yolo_detector PRIVATE /W4 /wd4819 /wd4267 /wd4244 /wd4100)

    # 设置运行时库
    if(CMAKE_BUILD_TYPE STREQUAL "Debug")
        set_target_properties(yolo_detector PROPERTIES
                MSVC_RUNTIME_LIBRARY "MultiThreadedDebugDLL"
        )
    else()
        set_target_properties(yolo_detector PROPERTIES
                MSVC_RUNTIME_LIBRARY "MultiThreadedDLL"
        )
    endif()
endif()

# ==================== 复制 DLL 文件 ====================
if(WIN32)
    # 1. 复制 ONNX Runtime DLL
    if(EXISTS "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll")
        add_custom_command(TARGET yolo_detector POST_BUILD
                COMMAND ${CMAKE_COMMAND} -E copy_if_different
                "${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll"
                "$<TARGET_FILE_DIR:yolo_detector>/onnxruntime.dll"
                COMMENT "复制 onnxruntime.dll"
        )
        message(STATUS "✓ 已配置复制 onnxruntime.dll")
    else()
        message(WARNING "⚠️ onnxruntime.dll 不存在: ${ONNXRUNTIME_LIB_DIR}/onnxruntime.dll")
    endif()

    # 2. 复制 OpenCV DLL
    # 查找 OpenCV 的 bin 目录
    if(OpenCV_BIN_DIR)
        set(OPENCV_BIN_DIR "${OpenCV_BIN_DIR}")
    elseif(EXISTS "${OpenCV_DIR}/bin")
        set(OPENCV_BIN_DIR "${OpenCV_DIR}/bin")
    elseif(EXISTS "${OpenCV_DIR}/x64/vc16/bin")
        set(OPENCV_BIN_DIR "${OpenCV_DIR}/x64/vc16/bin")
    else()
        # 尝试从库路径推断 bin 路径
        string(REPLACE "lib" "bin" OPENCV_BIN_DIR "${OpenCV_LIB_DIR}")
    endif()

    message(STATUS "OpenCV Bin 目录: ${OPENCV_BIN_DIR}")

    if(EXISTS "${OPENCV_BIN_DIR}")
        if(CMAKE_BUILD_TYPE STREQUAL "Debug")
            # 查找 Debug 版本的 DLL
            if(EXISTS "${OPENCV_BIN_DIR}/opencv_world4120d.dll")
                add_custom_command(TARGET yolo_detector POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${OPENCV_BIN_DIR}/opencv_world4120d.dll"
                        "$<TARGET_FILE_DIR:yolo_detector>/opencv_world4120d.dll"
                        COMMENT "复制 opencv_world4120d.dll"
                )
                message(STATUS "✓ 已配置复制 opencv_world4120d.dll")
            else()
                # 尝试复制所有 OpenCV DLL
                file(GLOB OPENCV_DLLS "${OPENCV_BIN_DIR}/*.dll")
                foreach(dll ${OPENCV_DLLS})
                    get_filename_component(DLL_NAME ${dll} NAME)
                    add_custom_command(TARGET yolo_detector POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${dll}"
                            "$<TARGET_FILE_DIR:yolo_detector>/${DLL_NAME}"
                            COMMENT "复制 ${DLL_NAME}"
                    )
                endforeach()
                message(STATUS "✓ 复制所有 OpenCV DLL")
            endif()
        else()
            # Release 版本
            if(EXISTS "${OPENCV_BIN_DIR}/opencv_world4120.dll")
                add_custom_command(TARGET yolo_detector POST_BUILD
                        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                        "${OPENCV_BIN_DIR}/opencv_world4120.dll"
                        "$<TARGET_FILE_DIR:yolo_detector>/opencv_world4120.dll"
                        COMMENT "复制 opencv_world4120.dll"
                )
                message(STATUS "✓ 已配置复制 opencv_world4120.dll")
            else()
                file(GLOB OPENCV_DLLS "${OPENCV_BIN_DIR}/*.dll")
                foreach(dll ${OPENCV_DLLS})
                    get_filename_component(DLL_NAME ${dll} NAME)
                    add_custom_command(TARGET yolo_detector POST_BUILD
                            COMMAND ${CMAKE_COMMAND} -E copy_if_different
                            "${dll}"
                            "$<TARGET_FILE_DIR:yolo_detector>/${DLL_NAME}"
                            COMMENT "复制 ${DLL_NAME}"
                    )
                endforeach()
                message(STATUS "✓ 复制所有 OpenCV DLL")
            endif()
        endif()
    else()
        message(WARNING "⚠️ OpenCV bin 目录不存在: ${OPENCV_BIN_DIR}")
    endif()
endif()

message(STATUS "\n========== 配置完成 ==========")
message(STATUS "项目: ${PROJECT_NAME}")
message(STATUS "OpenCV 版本: ${OpenCV_VERSION}")
message(STATUS "构建类型: ${CMAKE_BUILD_TYPE}")
message(STATUS "可执行文件: yolo_detector.exe")
message(STATUS "==============================\n")