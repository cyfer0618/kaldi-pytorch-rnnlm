// tensorflow-rnnlm.cc

// Copyright (C) 2017 Intellisist, Inc. (Author: Hainan Xu)

// See ../../COPYING for clarification regarding multiple authors
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//  http://www.apache.org/licenses/LICENSE-2.0
//
// THIS CODE IS PROVIDED *AS IS* BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, EITHER EXPRESS OR IMPLIED, INCLUDING WITHOUT LIMITATION ANY IMPLIED
// WARRANTIES OR CONDITIONS OF TITLE, FITNESS FOR A PARTICULAR PURPOSE,
// MERCHANTABLITY OR NON-INFRINGEMENT.
// See the Apache 2 License for the specific language governing permissions and
// limitations under the License.


#include <utility>
#include <fstream>

#include "pyrnnlm/pytorch-rnnlm.h"
#include "util/stl-utils.h"
#include "util/text-utils.h"

// torch::Tensorflow includes were moved after tfrnnlm/tensorflow-rnnlm.h include to
// avoid macro redefinitions. See also the note in tfrnnlm/tensorflow-rnnlm.h.
#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <memory>
#include <dirent.h>

namespace kaldi {
using std::ifstream;
using py_rnnlm::KaldiPyRnnlmWrapper;
using py_rnnlm::PyRnnlmDeterministicFst;
//using tensorflow::Status;

// read a unigram count file of the OOSs and generate extra OOS costs for words
void SetUnkPenalties(const string &filename,
                     const fst::SymbolTable& fst_word_symbols,
                     std::vector<float> *out) {
  if (filename == "")
    return;
  out->resize(fst_word_symbols.NumSymbols(), 0);  // default is 0
  ifstream ifile(filename.c_str());
  string word;
  float count, total_count = 0;
  while (ifile >> word >> count) {
    int id = fst_word_symbols.Find(word);
    KALDI_ASSERT(id != -1); // fst::kNoSymbol
    (*out)[id] = count;
    total_count += count;
  }

  for (int i = 0; i < out->size(); i++) {
    if ((*out)[i] != 0) {
      (*out)[i] = log ((*out)[i] / total_count);
    }
  }
}

// Read tensorflow checkpoint files
// Done ****
void KaldiPyRnnlmWrapper::ReadPyModel(const std::string &py_model_path,
                                      int32 num_threads) {

  // Need to initialise it
  // torch::jit::script::Module module;
  try {
    // Deserialize the ScriptModule from a file using torch::jit::load().
    std::cout << "Model " << py_model_path;
    // Load model in the module
    module = torch::jit::load(py_model_path+"/newmodel2.pt");
  }
  catch (const c10::Error& e) {
    std::cerr << "error loading the model\n";
    //return -1;
    return;
  }

  std::cout << "Language Model\n\n";

  // (Samrat): Think we need few of these, not all
  word_id_tensor_name_ = "word_id";
  context_tensor_name_ = "context";
  log_prob_tensor_name_ = "log_prob";
  rnn_out_tensor_name_ = "rnn_out";
  rnn_states_tensor_name_ = "rnn_states";
  initial_state_tensor_name_ = "initial_state";
  
}

// Done ****
// Batch_size = 1 they have hard code it
KaldiPyRnnlmWrapper::KaldiPyRnnlmWrapper(
    const KaldiPyRnnlmWrapperOpts &opts,
    const std::string &rnn_wordlist,
    const std::string &word_symbol_table_rxfilename,
    const std::string &unk_prob_file,
    const std::string &py_model_path): opts_(opts) {
  ReadPyModel(py_model_path, opts.num_threads);

  fst::SymbolTable *fst_word_symbols = NULL;
  if (!(fst_word_symbols =
        fst::SymbolTable::ReadText(word_symbol_table_rxfilename))) {
    KALDI_ERR << "Could not read symbol table from file "
        << word_symbol_table_rxfilename;
  }

  fst_label_to_word_.resize(fst_word_symbols->NumSymbols());

  for (int32 i = 0; i < fst_label_to_word_.size(); ++i) {
    fst_label_to_word_[i] = fst_word_symbols->Find(i);
    if (fst_label_to_word_[i] == "") {
      KALDI_ERR << "Could not find word for integer " << i << " in the word "
          << "symbol table, mismatched symbol table or you have discoutinuous "
          << "integers in your symbol table?";
    }
  }

  // first put all -1's; will check later
  fst_label_to_rnn_label_.resize(fst_word_symbols->NumSymbols(), -1);
  num_total_words = fst_word_symbols->NumSymbols();

  // read rnn wordlist and then generate ngram-label-to-rnn-label map
  oos_ = -1;
  { // input.
    ifstream ifile(rnn_wordlist.c_str());
    string word;
    int id = -1;
    eos_ = 0;
    while (ifile >> word) {
      id++;
      rnn_label_to_word_.push_back(word);  // vector[i] = word

      int fst_label = fst_word_symbols->Find(word);
      if (fst_label == -1) { // fst::kNoSymbol
        if (id == eos_)
          continue;

        KALDI_ASSERT(word == opts_.unk_symbol && oos_ == -1);
        oos_ = id;
        continue;
      }
      KALDI_ASSERT(fst_label >= 0);
      fst_label_to_rnn_label_[fst_label] = id;
    }
  }
  if (fst_label_to_word_.size() > rnn_label_to_word_.size()) {
    KALDI_ASSERT(oos_ != -1);
  }
  num_rnn_words = rnn_label_to_word_.size();

  // we must have an oos symbol in the wordlist
  if (oos_ == -1)
    return;

  for (int i = 0; i < fst_label_to_rnn_label_.size(); i++) {
    if (fst_label_to_rnn_label_[i] == -1) {
      fst_label_to_rnn_label_[i] = oos_;
    }
  }

  AcquireInitialTensors();
  SetUnkPenalties(unk_prob_file, *fst_word_symbols, &unk_costs_);
  delete fst_word_symbols;
}

KaldiPyRnnlmWrapper::~KaldiPyRnnlmWrapper() {
}
// Done
 
void KaldiPyRnnlmWrapper::AcquireInitialTensors() {
  // Status status;
  // get the initial context; this is basically the all-0 tensor
  /*
  (Samrat): Have to figure out get_initial_state(batch_size) ? what should btchsz be ?
  */
  //auto hidden = module.get_method("get_initial_state")({torch::tensor({1})});
  //initial_context_ = hidden.toTensor();

  initial_context_=module.get_method("get_initial_state")({torch::tensor({1})}).toTensor();


  //changed function call name (Samrat)
  auto bosword = torch::tensor({eos_});

  auto hidden = module.get_method("single_step_rnn_out")({initial_context_, bosword});
  initial_cell_ = hidden.toTensor();




  // {
  //   std::vector<torch::Tensor> state;
  //   status = bundle_.session->Run(std::vector<std::pair<string, torch::Tensor> >(),
  //                          {initial_state_tensor_name_}, {}, &state);
  //   if (!status.ok()) {
  //     KALDI_ERR << status.ToString();
  //   }
  //   initial_context_ = state[0];
  // }

  // get the initial pre-final-affine layer
  // {
  //   std::vector<torch::Tensor> state;
  //   torch::Tensor bosword(tensorflow::DT_INT32, {1, 1});
  //   bosword.scalar<int32>()() = eos_;  // eos_ is more like a sentence boundary

  //   std::vector<std::pair<string, torch::Tensor> > inputs = {
  //     {word_id_tensor_name_, bosword},
  //     {context_tensor_name_, initial_context_},
  //   };

  //   status = bundle_.session->Run(inputs, {rnn_out_tensor_name_}, {}, &state);
  //   if (!status.ok()) {
  //     KALDI_ERR << status.ToString();
  //   }
  //   initial_cell_ = state[0];
  // }
}


/*
// Need to change *****
BaseFloat KaldiPyRnnlmWrapper::GetLogProb(int32 word,
                                          int32 fst_word,
                                          const torch::Tensor &context_in,
                                          const torch::Tensor &cell_in,
                                          torch::Tensor *context_out,
                                          torch::Tensor *new_cell) {
  torch::Tensor thisword(torch::Tensor, {1, 1});

  thisword.scalar<int32>()() = word;

  std::vector<torch::Tensor> outputs;

  std::vector<std::pair<string, torch::Tensor> > inputs = {
    {word_id_tensor_name_, thisword},
    {context_tensor_name_, context_in},
  };

  if (context_out != NULL) {
    // The session will initialize the outputs
    // Run the session, evaluating our "c" operation from the graph
    Status status = bundle_.session->Run(inputs,
        {log_prob_tensor_name_,
         rnn_out_tensor_name_,
         rnn_states_tensor_name_}, {}, &outputs);
    if (!status.ok()) {
      KALDI_ERR << status.ToString();
    }

    *context_out = outputs[1];
    *new_cell = outputs[2];
  } else {
    // Run the session, evaluating our "c" operation from the graph
    Status status = bundle_.session->Run(inputs,
        {log_prob_tensor_name_}, {}, &outputs);
    if (!status.ok()) {
      KALDI_ERR << status.ToString();
    }
  }

  float ans;
  if (word != oos_) {
    ans = outputs[0].scalar<float>()();
  } else {
    if (unk_costs_.size() == 0) {
      ans = outputs[0].scalar<float>()() - log(num_total_words - num_rnn_words);
    } else {
      ans = outputs[0].scalar<float>()() + unk_costs_[fst_word];
    }
  }

  return ans;
}
*/

/*
  Below is my(Samrat) modified version of the above function only. 
  Replace if you think something is incorrect.
*/


BaseFloat KaldiPyRnnlmWrapper::GetLogProb(int32 word,
                                          int32 fst_word,
                                          const torch::Tensor &context_in,
                                          const torch::Tensor &cell_in,
                                          torch::Tensor *context_out,
                                          torch::Tensor *new_cell) {
  //torch::Tensor thisword(torch::Tensor, {1, 1});
  
  //thisword.scalar<int32>()() = word;
  torch::Tensor thisword = torch::tensor({word});


  //std::vector<torch::Tensor> outputs;

  // std::vector<std::pair<string, torch::Tensor> > inputs = {
  //   {word_id_tensor_name_, thisword},
  //   {context_tensor_name_, context_in},
  // };



  auto outputs = module.get_method("single_step")({context_in, thisword});
  if (context_out != NULL) {
    // The session will initialize the outputs
    // Run the session, evaluating our "c" operation from the graph
    // Status status = bundle_.session->Run(inputs,
    //     {log_prob_tensor_name_,
    //      rnn_out_tensor_name_,
    //      rnn_states_tensor_name_}, {}, &outputs);

    // if (!status.ok()) {
    //   KALDI_ERR << status.ToString();
    // }

    *context_out = module.get_method("single_step_rnn_out")({context_in, thisword}).toTensor();
    *new_cell = module.get_method("single_step_rnn_state")({context_in, thisword}).toTensor();
  } //else {
    // Run the session, evaluating our "c" operation from the graph
    // Status status = bundle_.session->Run(inputs,
    //     {log_prob_tensor_name_}, {}, &outputs);
    // if (!status.ok()) {
    //   KALDI_ERR << status.ToString();
    // }
  //}

  /*
    (Samrat): Can through error so have to check manually in testLM
    Hopefully expect it to return a float
  */
 
  float log_prob=(float)module.get_method("single_step_log")({context_in, thisword}).toDouble();
  float ans;
  if (word != oos_) {
    //ans = outputs[0].scalar<float>()();
    ans = log_prob;
  } else {
    if (unk_costs_.size() == 0) {
      //ans = outputs[0].scalar<float>()() - log(num_total_words - num_rnn_words);
      ans = log_prob - log(num_total_words - num_rnn_words);
    } else {
      //ans = outputs[0].scalar<float>()() + unk_costs_[fst_word];
      ans = log_prob + unk_costs_[fst_word];
    }
  }

  return ans;
}

// Done *****
const torch::Tensor& KaldiPyRnnlmWrapper::GetInitialContext() const {
  return initial_context_;
}

const torch::Tensor& KaldiPyRnnlmWrapper::GetInitialCell() const {
  return initial_cell_;
}

int KaldiPyRnnlmWrapper::FstLabelToRnnLabel(int i) const {
  KALDI_ASSERT(i >= 0 && i < fst_label_to_rnn_label_.size());
  return fst_label_to_rnn_label_[i];
}


// Done *****
PyRnnlmDeterministicFst::PyRnnlmDeterministicFst(int32 max_ngram_order,
                                             KaldiPyRnnlmWrapper *rnnlm) {
  KALDI_ASSERT(rnnlm != NULL);
  max_ngram_order_ = max_ngram_order;
  rnnlm_ = rnnlm;

  std::vector<Label> bos;
  const torch::Tensor& initial_context = rnnlm_->GetInitialContext();
  const torch::Tensor& initial_cell = rnnlm_->GetInitialCell();

  state_to_wseq_.push_back(bos);
  state_to_context_.push_back(new torch::Tensor(initial_context));
  state_to_cell_.push_back(new torch::Tensor(initial_cell));
  wseq_to_state_[bos] = 0;
  start_state_ = 0;
}

// Done *****
PyRnnlmDeterministicFst::~PyRnnlmDeterministicFst() {
  for (int i = 0; i < state_to_context_.size(); i++) {
    delete state_to_context_[i];
  }
  for (int i = 0; i < state_to_cell_.size(); i++) {
    delete state_to_cell_[i];
  }
}

// Done *****
void PyRnnlmDeterministicFst::Clear() {
  // similar to the destructor but we retain the 0-th entries in each map
  // which corresponds to the <bos> state
  for (int i = 1; i < state_to_context_.size(); i++) {
    delete state_to_context_[i];
  }
  for (int i = 1; i < state_to_cell_.size(); i++) {
    delete state_to_cell_[i];
  }

  state_to_context_.resize(1);
  state_to_cell_.resize(1);
  state_to_wseq_.resize(1);
  wseq_to_state_.clear();
  wseq_to_state_[state_to_wseq_[0]] = 0;
}

// Done *****
fst::StdArc::Weight PyRnnlmDeterministicFst::Final(StateId s) {
  // At this point, we should have created the state.
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  BaseFloat logprob = rnnlm_->GetLogProb(rnnlm_->GetEos(),
                         -1,  // only need type; this param will not be used
                         *state_to_context_[s],
                         *state_to_cell_[s], NULL, NULL);
  return Weight(-logprob);
}

// Done *****
bool PyRnnlmDeterministicFst::GetArc(StateId s, Label ilabel,
                                     fst::StdArc *oarc) {
  KALDI_ASSERT(static_cast<size_t>(s) < state_to_wseq_.size());

  std::vector<Label> wseq = state_to_wseq_[s];
  torch::Tensor *new_context = new torch::Tensor();
  torch::Tensor *new_cell = new torch::Tensor();

  // look-up the rnn label from the FST label
  int32 rnn_word = rnnlm_->FstLabelToRnnLabel(ilabel);
  BaseFloat logprob = rnnlm_->GetLogProb(rnn_word,
                                         ilabel,
                                         *state_to_context_[s],
                                         *state_to_cell_[s],
                                         new_context,
                                         new_cell);

  wseq.push_back(rnn_word);
  if (max_ngram_order_ > 0) {
    while (wseq.size() >= max_ngram_order_) {
      // History state has at most <max_ngram_order_> - 1 words in the state.
      wseq.erase(wseq.begin(), wseq.begin() + 1);
    }
  }

  std::pair<const std::vector<Label>, StateId> wseq_state_pair(
      wseq, static_cast<Label>(state_to_wseq_.size()));

  // Attemps to insert the current <lseq_state_pair>. If the pair already exists
  // then it returns false.
  typedef MapType::iterator IterType;
  std::pair<IterType, bool> result = wseq_to_state_.insert(wseq_state_pair);

  // If the pair was just inserted, then also add it to <state_to_wseq_> and
  // <state_to_context_>.
  if (result.second == true) {
    state_to_wseq_.push_back(wseq);
    state_to_context_.push_back(new_context);
    state_to_cell_.push_back(new_cell);
  } else {
    delete new_context;
    delete new_cell;
  }

  // Creates the arc.
  oarc->ilabel = ilabel;
  oarc->olabel = ilabel;
  oarc->nextstate = result.first->second;
  oarc->weight = Weight(-logprob);

  return true;
}

}  // namespace kaldi
