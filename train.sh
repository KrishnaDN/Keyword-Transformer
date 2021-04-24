#!/bin/bash

. ./path.sh || exit 1;
. ./cmd.sh || exit 1;


export CUDA_VISIBLE_DEVICES="0"
stage=0 # start from 0 if you need to start from data preparation
stop_stage=6
# data
nj=40
feat_dir=fbank
dict=data/dict/lang_char.txt
manifest=manifest
train_set=train_sp

cmvn=true
compress=true
fbank_conf=conf/fbank.conf
dir=exp/fbank_sp
checkpoint=

# use average_checkpoint will get better result

. utils/parse_options.sh || exit 1;

mkdir -p $manifest

 # Data preparation
utils/perturb_data_dir_speed.sh 0.9 data/train data/train_sp0.9
utils/perturb_data_dir_speed.sh 1.1 data/train data/train_sp1.1
utils/combine_data.sh data/train_sp data/train data/train_sp0.9 data/train_sp1.1

if [ ${stage} -le 1 ]; then
    # Feature extraction
    mkdir -p $feat_dir
    for x in ${train_set} test valid; do
        cp -r data/$x $feat_dir
        steps/make_fbank.sh --cmd "$train_cmd" --nj $nj \
            --write_utt2num_frames true --fbank_config $fbank_conf --compress $compress $feat_dir/$x
    done
    if $cmvn; then
        compute-cmvn-stats --binary=false scp:$feat_dir/$train_set/feats.scp \
            $feat_dir/$train_set/global_cmvn
    fi
fi



python format_data.py --feat_scp fbank/train_sp/feats.scp --text_file data/train_sp/text --cmvn_file fbank/train_sp/global_cmvn --store_folder /media/newhd/Google_Speech_Commands/features/train --manifest manifest/train
python format_data.py --feat_scp fbank/test/feats.scp --text_file data/test/text --cmvn_file fbank/train_sp/global_cmvn --store_folder /media/newhd/Google_Speech_Commands/features/test --manifest manifest/test
python format_data.py --feat_scp fbank/valid/feats.scp --text_file data/valid/text --cmvn_file fbank/train_sp/global_cmvn --store_folder /media/newhd/Google_Speech_Commands/features/valid --manifest manifest/valid
