# ======================================================================== #
# Copyright 2023 Dezeming                                                  #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

# ======================================================================== #
# Copyright 2018-2019 Ingo Wald                                            #
#                                                                          #
# Licensed under the Apache License, Version 2.0 (the "License");          #
# you may not use this file except in compliance with the License.         #
# You may obtain a copy of the License at                                  #
#                                                                          #
#     http://www.apache.org/licenses/LICENSE-2.0                           #
#                                                                          #
# Unless required by applicable law or agreed to in writing, software      #
# distributed under the License is distributed on an "AS IS" BASIS,        #
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. #
# See the License for the specific language governing permissions and      #
# limitations under the License.                                           #
# ======================================================================== #

if (POLICY CMP0048)
  cmake_policy(SET CMP0048 NEW)
endif (POLICY CMP0048)
if (POLICY CMP0077)
    cmake_policy(SET CMP0077 NEW)
endif()


project(NvidiaAIDenoiser VERSION 1.0.1)

cmake_minimum_required(VERSION 2.8)
if (NOT WIN32)
# visual studio doesn't like these (not need them):
set (CMAKE_CXX_FLAGS "--std=c++11")
set (CUDA_PROPAGATE_HOST_FLAGS ON)
endif()

# ------------------------------------------------------------------
# first, include gdt project to do some general configuration stuff
# (build modes, glut, optix, etc)
# ------------------------------------------------------------------
set(gdt_dir ${PROJECT_SOURCE_DIR}/common/gdt/)
set(CMAKE_MODULE_PATH ${CMAKE_MODULE_PATH} "${gdt_dir}/cmake/")
include(${gdt_dir}/cmake/configure_build_type.cmake)
include(${gdt_dir}/cmake/configure_optix.cmake)

#set(glfw_dir ${PROJECT_SOURCE_DIR}/submodules/glfw/)
#include(${gdt_dir}/cmake/configure_glfw.cmake)

mark_as_advanced(CUDA_SDK_ROOT_DIR)

# ------------------------------------------------------------------
# import gdt submodule
# ------------------------------------------------------------------
include_directories(${gdt_dir})
add_subdirectory(${gdt_dir} EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# build glfw
# ------------------------------------------------------------------
set(OpenGL_GL_PREFERENCE LEGACY)
if (WIN32)
#  set(glfw_dir ${PROJECT_SOURCE_DIR}/submodules/glfw/)
  set(glfw_dir ${PROJECT_SOURCE_DIR}/common/3rdParty/glfw/)
  include_directories(${glfw_dir}/include)
  add_subdirectory(${glfw_dir} EXCLUDE_FROM_ALL)
else()
  find_package(glfw3 REQUIRED)
endif()
include_directories(common)
add_subdirectory(common/glfWindow EXCLUDE_FROM_ALL)

# ------------------------------------------------------------------
# build opencv
# ------------------------------------------------------------------

#set(OpenCV_DIR "D:/DevTools/OpenCV/opencv-3.4.12-VS2019-Install/")

#set(OPENCV_INCLUDE
#  ${OpenCV_DIR}/include/
#)

#set(OPENCV_LIBS
#  ${OpenCV_DIR}/lib-Release/opencv_core3412.lib
#  ${OpenCV_DIR}/lib-Release/opencv_imgproc3412.lib
#  ${OpenCV_DIR}/lib-Release/opencv_imgcodecs3412.lib
#)

# ------------------------------------------------------------------
# build openexr
# ------------------------------------------------------------------

set(openexr_dir ${PROJECT_SOURCE_DIR}/common/openexr/)

set (ILMBASE_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
set (OPENEXR_NAMESPACE_VERSIONING OFF CACHE BOOL " " FORCE)
set (OPENEXR_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
set (ILMBASE_BUILD_SHARED_LIBS    OFF CACHE BOOL " " FORCE)
set (PYILMBASE_ENABLE             OFF CACHE BOOL " " FORCE)
set (OPENEXR_BUILD_UTILS          OFF CACHE BOOL " " FORCE)

add_subdirectory(${openexr_dir})

set_property (
TARGET IexMath IlmThread Half Iex Imath IlmImf HalfTest IexTest
IlmImfExamples IlmImfTest IlmImfUtil IlmImfUtilTest ImathTest
PROPERTY FOLDER "OpenEXR")

set (OPENEXR_INCLUDE
  ${CMAKE_CURRENT_SOURCE_DIR}/common/openexr/IlmBase/Imath
  ${CMAKE_CURRENT_SOURCE_DIR}/common/openexr/IlmBase/Half
  ${CMAKE_CURRENT_SOURCE_DIR}/common/openexr/IlmBase/Iex
  ${CMAKE_CURRENT_SOURCE_DIR}/common/openexr/OpenEXR/IlmImf
  ${CMAKE_CURRENT_BINARY_DIR}/common/openexr/IlmBase/config
  ${CMAKE_CURRENT_BINARY_DIR}/common/openexr/OpenEXR/config
)

set (OPENEXR_LIBS OpenEXR::IlmImf IlmBase::Imath IlmBase::Half)


# ------------------------------------------------------------------
# and final build rules for the project
# ------------------------------------------------------------------

set(optix_LIBRARY "")

# rendering in a glfwindow with continuous

find_package(OpenGL REQUIRED)

include_directories(
  ${OptiX_INCLUDE}
  ${OPENCV_INCLUDE}
  ${OPENEXR_INCLUDE}
)

cuda_add_library(toneMap
  Source/toneMap.cu)

add_executable(NvidiaAIDenoiser
  Source/optix7.h
  Source/CUDABuffer.h
  Source/SampleRenderer.h
  Source/SampleRenderer.cpp
  Source/DataInfo.h
  Source/exrLoad.h
  Source/exrLoad.cpp
  Source/exrSaver.h
  Source/exrSaver.cpp
  Source/exrHeader.h
  Source/main.cpp
  )

# set as startup item
set_property(DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR} PROPERTY VS_STARTUP_PROJECT NvidiaAIDenoiser)

set_target_properties(NvidiaAIDenoiser PROPERTIES LINK_FLAGS_RELEASE "/SUBSYSTEM:CONSOLE")
set_target_properties(NvidiaAIDenoiser PROPERTIES COMPILE_DEFINITIONS_RELEASE "_CONSOLE")

target_link_libraries(NvidiaAIDenoiser
  toneMap
  gdt
  # optix dependencies, for rendering
  ${optix_LIBRARY}
  ${CUDA_LIBRARIES}
  ${CUDA_CUDA_LIBRARY}
  # glfw and opengl, for display
  glfWindow
  glfw
  ${OPENGL_gl_LIBRARY}
  ${OPENCV_LIBS}
  ${OPENEXR_LIBS}
  )

# Copy test Files 
FILE(GLOB testfiles ${CMAKE_CURRENT_SOURCE_DIR}/TestFile/*.*)
file(COPY ${testfiles} DESTINATION ${CMAKE_CURRENT_BINARY_DIR}/TestFile/)




