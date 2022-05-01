# Introduction
This repository implements the CNN-based sentiment analysis of [Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) in Tensorflow 2.

# Requirements
```commandline
conda env create -f environment.yml
```

# Train

In this implementation, the words would firstly be converted to 300 dimensional vectors using pretrained word2vec of Google News 300 for transfer learning.
This transfering is very time-consuming and becomes the bottleneck of the training.
Therefore, it is necessary to firstly calculate the word vectors and save them to tfrecord.
Same as the paper, this implementation also uses 10-fold cross validation.
```commandline
export NEG_SAMPLE_PATH=<negative_sample_path> # change this to your negative sample file
export POS_SAMPLE_PATH=<positive_sample_path> # change this to your positive sample file
export TFRECORD_DIR=<tfrecord_path> # change this to your favorite directory
export KFOLD=10 # change this to any number
conda activate tensorflow
python data_preprocess.py --neg_sample_path $NEG_SAMPLE_PATH\
                          --pos_sample_path $POS_SAMPLE_PATH\
                          --save_dir $TFRECORD_DIR
                          --k_fold $KFOLD
```
Generated tfrecord files would be saved in **TFRECORD_DIR**.

The train.py would search .tfrecord files in the **TFRECORD_DIR** and perform k-fold cross validation.
k is the number of tfrecords in the directory.
```
export CKPT_DIR=<checkpoint_dir> # change this to your favorite directory to save the trained weights
conda activate tensorflow
python train.py --record_dir $TFRECORD_DIR\
                --ckpt_dir $CKPT_DIR
```

# Performance
Since the paper didn't provide enough details of the implementation, the performance is about 2% lower than the reported accuracy.  
10-fold cross validation on MR: 79.01%   
