cmake_minimum_required(VERSION 3.5)
project(CartPoleAppli)

set(CMAKE_AUTOMOC ON)
set(CMAKE_AUTOUIC ON)
set(CMAKE_CXX_STANDARD 17)

find_package(Qt5 REQUIRED COMPONENTS Widgets)

add_executable(CartPoleAppli
    main.cpp
    Agent.cpp
    Agent.h
    SimpleNN.cpp
    SimpleNN.h
    Widget.cpp
    Widget.h
)

target_link_libraries(CartPoleAppli Qt5::Widgets)

