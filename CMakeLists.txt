cmake_minimum_required(VERSION 3.15...3.26)
project(pyipopt_project LANGUAGES CXX)

# 关闭烦人的警告
set(PYBIND11_FINDPYTHON ON)
find_package(pybind11 CONFIG REQUIRED)


# 优先读取云端环境变量，如果没有再用本地的 D 盘路径
set(ENV_IPOPT_DIR $ENV{IPOPT_DIR})
if(ENV_IPOPT_DIR)
    set(IPOPT_DIR ${ENV_IPOPT_DIR} CACHE PATH "Path to IPOPT directory")
    message(STATUS "☁️ 检测到环境变量，使用 IPOPT 路径: ${IPOPT_DIR}")
else()
    set(IPOPT_DIR "D:/microsoft_downloads/Ipopt-3.14.19-win64-msvs2022-md/Ipopt-3.14.19-win64-msvs2022-md" CACHE PATH "Path to IPOPT directory")
    message(STATUS "💻 使用本地路径: ${IPOPT_DIR}")
endif()

# set(IPOPT_DIR "D:/microsoft_downloads/Ipopt-3.14.19-win64-msvs2022-md/Ipopt-3.14.19-win64-msvs2022-md" CACHE PATH "Path to IPOPT directory")

# 🚨 修正点 1：把末尾的 /coin-or 删掉，防止头文件路径双层嵌套
set(IPOPT_INCLUDE_DIR "${IPOPT_DIR}/include")
set(IPOPT_LIB_DIR "${IPOPT_DIR}/lib")

# 2. 精准寻找 DLL 的导入库
find_library(IPOPT_LIB NAMES ipopt.dll.lib ipopt.dll PATHS ${IPOPT_LIB_DIR} REQUIRED)

pybind11_add_module(pyipopt src/wrapper.cpp)

target_include_directories(pyipopt PRIVATE ${IPOPT_INCLUDE_DIR})
target_link_libraries(pyipopt PRIVATE ${IPOPT_LIB})

# 在 MSVC 选项里加入 /utf-8，彻底解决中文注释报错问题
if(MSVC)
    target_compile_options(pyipopt PRIVATE /O2 /fp:fast /utf-8)
else()
    target_compile_options(pyipopt PRIVATE -O3 -ffast-math)
endif()

# 🚨 新增这一行：告诉 CMake 把编译好的 pyipopt.pyd 文件塞进安装包的根目录！
install(TARGETS pyipopt DESTINATION .)