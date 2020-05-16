cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
set(CMAKE_VERBOSE_MAKEFILE ON)
set(PUBLIC_HEADERS
    pytorch-rnnlm.h
)

add_library(kaldi-pyrnnlm
    pytorch-rnnlm.cc
)

#set(D_GLIBCXX_USE_CXX11_ABI,"0")

target_include_directories(kaldi-pyrnnlm PUBLIC $<BUILD_INTERFACE:/home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch/include/torch/csrc/api/include>
PUBLIC $<BUILD_INTERFACE:/home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch/include>
PUBLIC $<BUILD_INTERFACE:/home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch/include/caffe2>
PUBLIC $<BUILD_INTERFACE:/home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch/include/ATen/core>
PUBLIC $<BUILD_INTERFACE:/home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch/include/c10/util>
PUBLIC $<BUILD_INTERFACE:/home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch/lib>
PUBLIC $<BUILD_INTERFACE:/usr/local/lib/python3.6/dist-packages/torch/lib>
#PUBLIC $<BUILD_INTERFACE:/usr/local/cuda/lib64>
#PUBLIC $<BUILD_INTERFACE:/usr/local/cuda/lib64/stubs>
)

#link_directories(/home/rakesh/rishabh_workspace/ASR/kaldi_old/tools/libtorch)

#target_include_directories(kaldi-pyrnnlm PUBLIC 
#    ${PYTORCH_DIR}
#    ${PYTORCH_DIR}/bin
#    ${PYTORCH_DIR}/include
#    ${PYTORCH_DIR}/share
#
#)

#set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED PATHS /home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch)

target_include_directories(kaldi-pyrnnlm PUBLIC 
     $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/..>
     $<INSTALL_INTERFACE:include/kaldi>
     #/home/rakesh/rishabh_workspace/ASR/kaldi_old/tools/libtorch/include
     #/home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch/include/torch/csrc/api/include
)

target_compile_options(kaldi-pyrnnlm PUBLIC 
    -lz
    -ldl
    -fPIC
)

target_link_libraries(kaldi-pyrnnlm PUBLIC
    kaldi-lm
    kaldi-util
    kaldi-matrix
    kaldi-base
    libc10.so
    libc10_cuda.so
    #-lz
    #-ldl
    #-fPIC
    #${TORCH_LIBRARIES}
    #-L/home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch
    #-L/home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch/lib
    #-L/home/rakesh/rishabh_workspace/Garbage/kaldi/tools/cuda/lib64

)



#find_package(Torch REQUIRED PATHS /home/rakesh/rishabh_workspace/Garbage/kaldi/tools/libtorch)

MESSAGE("TORCH LIBS ${TORCH_LIBRARIES}")

#target_link_libraries(kaldi-pyrnnlm "${TORCH_LIBRARIES}")
#set_property(TARGET kaldi-pyrnnlm PROPERTY CXX_STANDARD 14)

install(TARGETS kaldi-pyrnnlm
    EXPORT kaldi-targets
    ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
)

install(FILES ${PUBLIC_HEADERS} DESTINATION include/kaldi/pyrnnlm)

