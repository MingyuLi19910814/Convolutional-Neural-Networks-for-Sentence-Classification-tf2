import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import argparse
import data_helpers
import numpy as np
import pathlib
import random
from gensim import downloader
import re
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument('--neg_sample_path', type=str, default='data/rt-polaritydata/rt-polarity.neg')
parser.add_argument('--pos_sample_path', type=str, default='data/rt-polaritydata/rt-polarity.pos')
parser.add_argument('--save_path', type=str, default='tfrecord/rt-polarity.tfrecord')
parser.add_argument('--max_document_len', type=int, default=-1)

tokenizer = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)

def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

def encode_sentence(sentence, label, vocabulary, max_document_len):
    '''
    :param sentence:  sentence of str type
    :param label: one-hot label
    :param vocabulary:  google news 300
    :param max_document_len: maximum word number in a sentence
    :return:
    '''
    encoded_feature = np.zeros(shape=(max_document_len, 300), dtype=np.float32)
    for idx, word in enumerate(tokenizer.findall(sentence)):
        if idx >= max_document_len:
            break
        if word in vocabulary:
            encoded_feature[idx] = vocabulary[word]
    feature = {
        'feature': _bytes_feature(encoded_feature.tobytes()),
        'label': _bytes_feature(label.tobytes())
    }
    msg = tf.train.Example(features=tf.train.Features(feature=feature))
    return msg

def main(args):
    assert os.path.isfile(args.neg_sample_path) and os.path.isfile(args.pos_sample_path)
    sentences, labels = data_helpers.load_data_and_labels(args.pos_sample_path, args.neg_sample_path)
    max_document_len = args.max_document_len if args.max_document_len != -1 else max([len(x.split(" ")) for x in sentences])

    #shuffle the data
    data = list(zip(sentences, labels))
    random.shuffle(data)
    sentences, labels = zip(*data)

    #load google news 300 vocabulary
    vocabulary = downloader.load('word2vec-google-news-300')

    #encode
    save_dir = pathlib.Path(args.save_path).parent
    os.makedirs(save_dir, exist_ok=True)
    with tf.io.TFRecordWriter(args.save_path) as writer:
        for sentence, label in tqdm(zip(sentences, labels)):
            msg = encode_sentence(sentence, label, vocabulary, max_document_len)
            writer.write(msg.SerializeToString())

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)