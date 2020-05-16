cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)
#find_package(Torch REQUIRED)
#find_package(Torch REQUIRED PATHS /home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch)

set(CMAKE_CXX_STANDARD 14)

add_kaldi_executable(NAME lattice-lmrescore-py-rnnlm SOURCES lattice-lmrescore-py-rnnlm.cc DEPENDS kaldi-pyrnnlm kaldi-lat)

#target_link_libraries(lattice-lmrescore-py-rnnlm "${PYTORCH_DIR}")
#set_property(TARGET lattice-lmrescore-py-rnnlm PROPERTY CXX_STANDARD 14)
