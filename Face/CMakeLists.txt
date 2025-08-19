cmake_minimum_required(VERSION 3.10)
project(rpi_facial_parts_monitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(Qt5 REQUIRED COMPONENTS Core)
find_package(OpenCV REQUIRED)

add_executable(rpi_facial_parts_monitor main.cpp)
target_link_libraries(rpi_facial_parts_monitor Qt5::Core ${OpenCV_LIBS})

find_package(Qt5 REQUIRED COMPONENTS Core WebSockets)

add_executable(ws_server ws_server.cpp)
target_link_libraries(ws_server Qt5::Core Qt5::WebSockets)
