# we assume KALDI_ROOT is already defined
[ -z "$KALDI_ROOT" ] && echo >&2 "The variable KALDI_ROOT must be already defined" && exit 1
# The formatting of the path export command is intentionally weird, because
# this allows for easy diff'ing
export PATH=\
${KALDI_ROOT}/build/src/bin:\
${KALDI_ROOT}/build/src/chainbin:\
${KALDI_ROOT}/build/src/featbin:\
${KALDI_ROOT}/build/src/fgmmbin:\
${KALDI_ROOT}/build/src/fstbin:\
${KALDI_ROOT}/build/src/gmmbin:\
${KALDI_ROOT}/build/src/ivectorbin:\
${KALDI_ROOT}/build/src/kwsbin:\
${KALDI_ROOT}/build/src/latbin:\
${KALDI_ROOT}/build/src/lmbin:\
${KALDI_ROOT}/build/src/nnet2bin:\
${KALDI_ROOT}/build/src/nnet3bin:\
${KALDI_ROOT}/build/src/nnetbin:\
${KALDI_ROOT}/build/src/online2bin:\
${KALDI_ROOT}/build/src/onlinebin:\
${KALDI_ROOT}/build/src/rnnlmbin:\
${KALDI_ROOT}/build/src/sgmm2bin:\
${KALDI_ROOT}/build/src/sgmmbin:\
${KALDI_ROOT}/build/src/pyrnnlmbin:\
${KALDI_ROOT}/build/src/tfrnnlmbin:\
${KALDI_ROOT}/build/src/cudadecoderbin:\
$PATH
