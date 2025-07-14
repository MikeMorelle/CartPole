cmake_minimum_required(VERSION 3.5)
project(CartPoleApp)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Qt5 REQUIRED COMPONENTS Widgets)

add_executable(CartPoleApp
    main.cpp
    CartPoleWidget.cpp
    CartPoleWidget.h
    CartPoleSim.cpp
    CartPoleSim.h
)

target_link_libraries(CartPoleApp Qt5::Widgets)

