cmake_minimum_required(VERSION 3.10)
project(rpi_facial_parts_monitor LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Qt (Core + Network 한 번에)
find_package(Qt5 REQUIRED COMPONENTS Core Network)

# OpenCV
find_package(OpenCV REQUIRED)

# moc 자동 처리 (Q_OBJECT 쓰는 파일이 있을 때 안전)
set(CMAKE_AUTOMOC ON)

add_executable(rpi_facial_parts_monitor main.cpp)

target_link_libraries(rpi_facial_parts_monitor
    PRIVATE
        Qt5::Core
        Qt5::Network
        ${OpenCV_LIBS}
)
