import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import time
import datetime
import data_helpers
import argparse
import preprocessing
import text_cnn
from gensim import downloader
from sklearn.model_selection import train_test_split
import re


def create_wordvec_dataset(X, Y, vocabulary, batch_size, max_document_len):
    tokenizer = re.compile(r"[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\'\w\-]+", re.UNICODE)
    def word2vec_process(x, y):
        word_vecs = np.zeros(shape=(max_document_len, 300), dtype=np.float32)
        x = x.numpy().decode('utf-8')
        for idx, word in enumerate(tokenizer.findall(x)):
            if idx >= max_document_len:
                break
            if word in vocabulary:
                word_vecs[idx] = vocabulary[word]
        return word_vecs, tf.cast(y, tf.float32)
    dataset = tf.data.Dataset.from_tensor_slices((X, Y))
    dataset = dataset.map(lambda x, y: tf.py_function(word2vec_process, inp=[x, y], Tout=[np.float32, np.float32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)
    return dataset

def preprocess(args):
    X, Y = data_helpers.load_data_and_labels(args.positive_data_file, args.negative_data_file)
    max_document_length = max([len(x.split(" ")) for x in X])
    train_x, val_x, train_y, val_y = train_test_split(X, Y, test_size=args.dev_sample_percentage)
    print('loading Google News 300 Word2vec...')
    google_vocabulary = downloader.load('word2vec-google-news-300')
    print('creating datasets')
    train_ds = create_wordvec_dataset(train_x, train_y, google_vocabulary, args.batch_size, max_document_length)
    val_ds = create_wordvec_dataset(val_x, val_y, google_vocabulary, args.batch_size, max_document_length)
    return train_ds, val_ds

def create_dataset(x, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.batch(batch_size)
    return dataset


def train(train_ds, val_ds, args):
    tf.config.set_soft_device_placement(True)
    batch_x, batch_y = next(iter(train_ds))
    model = text_cnn.create_text_cnn_pretrained(sequence_length=batch_x.shape[1],
                                                num_classes=batch_y.shape[1],
                                                embedding_size=300,
                                                filter_sizes=list(map(int, args.filter_sizes.split(","))),
                                                num_filters=args.num_filters,
                                                dropout_keep_prob=args.dropout_keep_prob,
                                                l2_reg_lambda=args.l2_reg_lambda)
    model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['categorical_accuracy'])
    model.fit(train_ds, epochs=args.num_epochs, validation_data=val_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--learning_rate', type=float, default=1e-3, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch Size (default: 64)")
    parser.add_argument('--num_epochs', type=int, default=30, help="Number of training epochs (default: 200)")
    parser.add_argument('--use_google_news_300', type=bool, default=True, help="if use word2vec vocabulary pretrained on google news")
    parser.add_argument('--dev_sample_percentage', type=float, default=0.1, help="Percentage of the training data to use for validation")
    parser.add_argument('--positive_data_file', type=str, default='./data/rt-polaritydata/rt-polarity.pos', help='Data source for the positive data.')
    parser.add_argument('--negative_data_file', type=str, default='./data/rt-polaritydata/rt-polarity.neg', help='Data source for the negative data.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimensionality of character embedding (default: 128)')
    parser.add_argument('--filter_sizes', type=str, default="3", help='Comma-separated filter sizes (default: "3,4,5")')
    parser.add_argument('--num_filters', type=int, default=128, help='Number of filters per filter size (default: 128)')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help="Dropout keep probability (default: 0.5)")
    parser.add_argument('--l2_reg_lambda', type=float, default=0.0, help="L2 regularization lambda (default: 0.0)")
    args = parser.parse_args()
    train_ds, val_ds = preprocess(args)
    train(train_ds, val_ds, args)