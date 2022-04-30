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

def preprocess(args):
    x_text, y = data_helpers.load_data_and_labels(args.positive_data_file, args.negative_data_file)

    # Build vocabulary
    max_document_length = max([len(x.split(" ")) for x in x_text])
    vocab_processor = preprocessing.VocabularyProcessor(max_document_length)
    x = np.array(list(vocab_processor.fit_transform(x_text)))
    # Randomly shuffle data
    np.random.seed(10)
    shuffle_indices = np.random.permutation(np.arange(len(y)))
    x_shuffled = x[shuffle_indices]
    y_shuffled = y[shuffle_indices]

    # Split train/test set
    # TODO: This is very crude, should use cross-validation
    dev_sample_index = -1 * int(args.dev_sample_percentage * float(len(y)))
    x_train, x_dev = x_shuffled[:dev_sample_index], x_shuffled[dev_sample_index:]
    y_train, y_dev = y_shuffled[:dev_sample_index], y_shuffled[dev_sample_index:]
    print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))
    print("Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))
    return x_train, y_train, vocab_processor, x_dev, y_dev

def create_dataset(x, y, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((x, y))
    dataset = dataset.shuffle(buffer_size=10000).batch(batch_size)
    return dataset


def train(x_train, y_train, vocab_processor, x_dev, y_dev, args):
    tf.config.set_soft_device_placement(True)
    train_ds = create_dataset(x_train, y_train, args.batch_size)
    val_ds = create_dataset(x_dev, y_dev, args.batch_size)
    model = text_cnn.create_text_cnn(sequence_length=x_train.shape[1],
                                     num_classes=y_train.shape[1],
                                     vocab_size=len(vocab_processor.vocabulary_),
                                     embedding_size=args.embedding_dim,
                                     filter_sizes=list(map(int, args.filter_sizes.split(","))),
                                     num_filters=args.num_filters,
                                     dropout_keep_prob=args.dropout_keep_prob,
                                     l2_reg_lambda=args.l2_reg_lambda)
    model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
                  loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                  metrics=['accuracy'])
    model.fit(train_ds, epochs=args.num_epochs, validation_data=val_ds)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--dev_sample_percentage', type=float, default=0.1, help="Percentage of the training data to use for validation")
    parser.add_argument('--positive_data_file', type=str, default='./data/rt-polaritydata/rt-polarity.pos', help='Data source for the positive data.')
    parser.add_argument('--negative_data_file', type=str, default='./data/rt-polaritydata/rt-polarity.neg', help='Data source for the negative data.')
    parser.add_argument('--embedding_dim', type=int, default=128, help='Dimensionality of character embedding (default: 128)')
    parser.add_argument('--filter_sizes', type=str, default="3", help='Comma-separated filter sizes (default: "3,4,5")')
    parser.add_argument('--num_filters', type=int, default=128, help='Number of filters per filter size (default: 128)')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help="Dropout keep probability (default: 0.5)")
    parser.add_argument('--l2_reg_lambda', type=float, default=0.0, help="L2 regularization lambda (default: 0.0)")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch Size (default: 64)")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument('--evaluate_every', type=int, default=100, help="Evaluate model on dev set after this many steps (default: 100)")
    parser.add_argument('--checkpoint_every', type=int, default=100, help="Save model after this many steps (default: 100)")
    parser.add_argument('--num_checkpoints', type=int, default=5, help="Number of checkpoints to store (default: 5)")
    args = parser.parse_args()
    x_train, y_train, vocab_processor, x_dev, y_dev = preprocess(args)
    train(x_train, y_train, vocab_processor, x_dev, y_dev, args)