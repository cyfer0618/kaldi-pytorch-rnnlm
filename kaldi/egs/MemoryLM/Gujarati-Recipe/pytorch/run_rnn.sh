#!/bin/bash
cmd=run.pl
ngram_order=4 # this option when used, the rescoring binary makes an approximation
    # to merge the states of the FST generated from RNNLM. e.g. if ngram-order = 4
    # then any history that shares last 3 words would be merged into one state
stage=1
weight=0.5   # when we do lattice-rescoring, instead of replacing the lm-weights
    # in the lattice with RNNLM weights, we usually do a linear combination of
    # the 2 and the $weight variable indicates the weight for the RNNLM scores

. ./utils/parse_options.sh
. ./cmd.sh
. ./path.sh

set -e

dir=data/pytorch
mkdir -p $dir



# Need to install pytorch (torch) Manually
echo "Things to do : "
echo "1. Need to install pytorch (torch) Manually at tools/torch"
echo "2. Save the model at data/pytorch/rnnlm Manually"


# if [ $stage -le 1 ]; then
#   pytorch/rnnlm_data_prep.sh $dir
# fi

# mkdir -p $dir
# if [ $stage -le 2 ]; then
# # the following script uses TensorFlow. You could use tools/extras/install_tensorflow_py.sh to install it
#   $cuda_cmd $dir/train_rnnlm.log utils/parallel/limit_num_gpus.sh \
#     python steps/tfrnnlm/lstm.py --data_path=$dir --save_path=$dir/rnnlm --vocab_path=$dir/wordlist.rnn.final
# fi

# Need to save the Model at data/pytorch/rnnlm


#Change the LM from this to our LM
#final_lm=ami_fsh.o3g.kn
#LM=$final_lm.pr1-7

echo " e.g.: $0 data/lang_test_tg data/pytorch data/test exp/chain/tdnn1g_sp/decode_tg_test exp/chain/test_pyrnnlm"
echo "Usage: $0 [options] <old-lang-dir> <rnnlm-dir> <data-dir> <input-decode-dir> <output-decode-dir>"

if [ $stage -le 3 ]; then
  for decode_set in tg_test tgpr_test bd_tgpr_test; do
    basedir=exp/chain/tdnn1g_sp
    decode_dir=${basedir}/decode_${decode_set}

    
    # For cmd -> "$train_cmd" {run.pl}
    # old-lang-dir -> data/lang_test_tg
    # Lattice rescoring

    pytorch/lmrescore_rnnlm_lat.sh \
      --cmd "$train_cmd" \
      --weight $weight --max-ngram-order $ngram_order \
      data/lang_test_tg $dir \
      data/test ${decode_dir} \
      ${decode_dir}.pyrnnlm.lat.${ngram_order}gram.$weight  &

  done
fi

wait
