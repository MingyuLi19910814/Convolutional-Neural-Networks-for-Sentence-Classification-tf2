import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import numpy as np
import argparse
import text_cnn
from glob import glob
import shutil

def decode_tfrecord(record_bytes):
    feature = tf.io.parse_single_example(
        record_bytes,
        {"feature": tf.io.FixedLenFeature([], dtype=tf.string),
         "label": tf.io.FixedLenFeature([], dtype=tf.string),
         'd1': tf.io.FixedLenFeature([], dtype=tf.int64),
         'd2': tf.io.FixedLenFeature([], dtype=tf.int64)}
    )
    x = tf.io.decode_raw(feature['feature'], tf.float32)
    x = tf.reshape(x, (feature['d1'], feature['d2'], 1))
    y = tf.io.decode_raw(feature['label'], tf.int64)
    return x, y

def dataset_from_tfrecord(tfrecord_files, args):
    dataset = tf.data.TFRecordDataset(tfrecord_files).map(decode_tfrecord, num_parallel_calls=6).shuffle(1000).batch(args.batch_size)
    return dataset

def train(args):
    assert os.path.isdir(args.record_dir)
    shutil.rmtree(args.ckpt_dir)
    os.makedirs(args.ckpt_dir)
    tfrecord_files = glob(os.path.join(args.record_dir, '*.tfrecord'))
    tfrecord_files.sort()
    test_acc = []
    print('{} tfrecord files found'.format(tfrecord_files.__len__()))
    if args.k_fold:
        num_fold = tfrecord_files.__len__()
        print('Perform {}-fold cross validation'.format(num_fold))
        for i_fold in range(num_fold):
            print('*' * 30 + "start fold {}".format(i_fold) + '*' * 30)
            # prepare train, validate, test datasets
            val_idx = i_fold
            test_idx = (i_fold + 1) % num_fold
            val_records = [tfrecord_files[val_idx]]
            test_records = [tfrecord_files[test_idx]]
            train_records = [tfrecord_files[i] for i in range(num_fold) if i != val_idx and i != test_idx]
            train_ds = dataset_from_tfrecord(train_records, args)
            val_ds = dataset_from_tfrecord(val_records, args)
            test_ds = dataset_from_tfrecord(test_records, args)
            batch_x, batch_y = next(iter(train_ds))
            # save best model
            ckpt_path = os.path.join(args.ckpt_dir, str(i_fold))
            ckpt_callback = tf.keras.callbacks.ModelCheckpoint(filepath=ckpt_path,
                                                               save_weights_only=True,
                                                               monitor='val_loss',
                                                               mode='min',
                                                               save_best_only=True)

            model = text_cnn.create_text_cnn_pretrained(sequence_length=batch_x.shape[1],
                                                        num_classes=batch_y.shape[1],
                                                        embedding_size=300,
                                                        filter_sizes=list(map(int, args.filter_sizes.split(","))),
                                                        num_filters=args.num_filters,
                                                        dropout_keep_prob=args.dropout_keep_prob,
                                                        l2_reg_lambda=args.l2_reg_lambda)
            model.compile(optimizer=tf.keras.optimizers.Adam(args.learning_rate, clipnorm=args.l2_norm_clip),
                          loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
                          metrics=['categorical_accuracy'])
            model.fit(train_ds, epochs=args.num_epochs, validation_data=val_ds, callbacks=[ckpt_callback])
            model.load_weights(ckpt_path)
            res = model.evaluate(test_ds)
            test_acc.append(res[1])
        print('overall test accuracy = {}'.format(np.mean(test_acc)))



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--record_dir', type=str, default='tfrecord/rt-polarity', help="path to the tfrecords of the dataset")
    parser.add_argument('--ckpt_dir', type=str, default='checkpoint')
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="learning rate")
    parser.add_argument('--batch_size', type=int, default=512, help="Batch Size (default: 64)")
    parser.add_argument('--num_epochs', type=int, default=200, help="Number of training epochs (default: 200)")
    parser.add_argument('--filter_sizes', type=str, default="3,4,5", help='Comma-separated filter sizes (default: "3,4,5")')
    parser.add_argument('--num_filters', type=int, default=100, help='Number of filters per filter size (default: 128)')
    parser.add_argument('--dropout_keep_prob', type=float, default=0.5, help="Dropout keep probability (default: 0.5)")
    parser.add_argument('--l2_reg_lambda', type=float, default=0.1, help="L2 regularization lambda (default: 0.0)")
    parser.add_argument('--l2_norm_clip', type=float, default=3, help="constrain of l2-norm of the weight vectors")
    args = parser.parse_args()
    train(args)
