# This directory is not compiled automatically when you do make under src/
# to compile the TensorFlow related code,

# you would need to run tools/extras/install_tensorflow_cc.sh first to generate
# the static library files required for compilation, and then

# cd ${KALDI_ROOT}/src
# cd tfrnnlm
# make
# cd ../tfrnnlmbin/
# make

include ../kaldi.mk

PYTORCH = ../../tools/libtorch

all:

EXTRA_CXXFLAGS = -Wno-sign-compare \
                 -I${PYTORCH}

OBJFILES = pytorch-rnnlm.o

TESTFILES =

LIBNAME = kaldi-pytorch-rnnlm

ADDLIBS = ../lm/kaldi-lm.a ../util/kaldi-util.a ../matrix/kaldi-matrix.a \
          ../base/kaldi-base.a
LDLIBS +=  -lz -ldl -fPIC -lrt

include ../makefiles/default_rules.mk
