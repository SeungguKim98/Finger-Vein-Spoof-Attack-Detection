""" USAGE
python ./train.py --train_src_dir ./datasets/horse2zebra/trainA --train_tar_dir ./datasets/horse2zebra/trainB --test_src_dir ./datasets/horse2zebra/testA --test_tar_dir ./datasets/horse2zebra/testB
"""

import os
import argparse
import datetime
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import random
from sam import SAM
import glob
from PIL import Image
from modules.sam_cut_model import CUT_model
from utils import create_dir, load_image, test_load_image


def ArgParse():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='CUT training usage.')
    # Training
    parser.add_argument('--mode', help="Model's mode be one of: 'cut', 'fastcut'", type=str, default='cut',
                        choices=['cut', 'fastcut'])
    parser.add_argument('--epochs', help='Number of training epochs', type=int, default=400)
    parser.add_argument('--batch_size', help='Training batch size', type=int, default=1)
    parser.add_argument('--beta_1', help='First Momentum term of adam', type=float, default=0.5)
    parser.add_argument('--beta_2', help='Second Momentum term of adam', type=float, default=0.999)
    parser.add_argument('--lr', help='Initial learning rate for adam', type=float, default=0.0002)
    parser.add_argument('--lr_decay_rate', help='lr_decay_rate', type=float, default=0.9)
    parser.add_argument('--lr_decay_step', help='lr_decay_step', type=int, default=100000)
    # Define data
    parser.add_argument('--out_dir', help='Outputs folder', type=str, default='save path')
    parser.add_argument('--train_dir', help='Train-source dataset folder', type=str, default='')
    parser.add_argument('--test_dir', help='Test-source dataset folder', type=str, default='test images path')
    # Misc
    parser.add_argument('--ckpt', help='Resume training from checkpoint', type=str)
    parser.add_argument('--save_n_epoch', help='Every n epochs to save checkpoints', type=int, default=1)
    parser.add_argument('--impl', help="(Faster)Custom op use:'cuda'; (Slower)Tensorflow op use:'ref'", type=str,
                        default='ref', choices=['ref', 'cuda'])

    args = parser.parse_args()

    # Check arguments
    assert args.lr > 0
    assert args.epochs > 0
    assert args.batch_size > 0
    assert args.save_n_epoch > 0
    assert os.path.exists(args.train_dir), 'Error: Train source dataset does not exist.'
    assert os.path.exists(args.test_dir), 'Error: Test source dataset does not exist.'

    return args


def main(args):
    # Create datasets
    train_dataset, test_dataset, test_dataset_name = create_dataset(args.train_dir,
                                                     args.test_dir,
                                                     args.batch_size)

    # Get image shape
    source_image, target_image = next(iter(train_dataset))
    source_shape = source_image.shape[1:]
    target_shape = target_image.shape[1:]

    # Create model
    cut = CUT_model(source_shape, target_shape, cut_mode=args.mode, impl=args.impl)

    for save_epoch in range(1, 401):
        cut.load_weights(os.path.join(args.out_dir, 'checkpoints/') + str(save_epoch).zfill(3)).assert_existing_objects_matched()

        for i, (source, target) in enumerate(test_dataset.take(len(test_dataset))):
            translated = cut.netG(source)[0].numpy()
            translated_img = tensor_to_image(translated)
            save_path = os.path.join(args.out_dir, 'save_img') + "/" + str(save_epoch)
            os.makedirs(save_path, exist_ok=True)
            translated_img.save(save_path + "/" + test_dataset_name[i])

def tensor_to_image(tensor):
    tensor = tensor*0.5+0.5
    tensor = tensor*255
    tensor = np.array(tensor, dtype=np.uint8)
    if np.ndim(tensor)>3:
        assert tensor.shape[0] == 1
        tensor = tensor[0]
    return Image.fromarray(tensor)


# 데이터셋 만들기 (input + target(intra-class))
def list_compare(src_dataset, tar_dataset):
    for i in range(len(src_dataset)):
        if src_dataset[i] == tar_dataset[i]:
            return True
    return False


def interclassshuffle(dataset_path):
    src_dataset = glob.glob(os.path.join(dataset_path, '*.bmp'))
    tar_dataset = glob.glob(os.path.join(dataset_path, '*.bmp'))
    random.shuffle(tar_dataset)
    bool_shuffle = list_compare(src_dataset, tar_dataset)
    while bool_shuffle:
        random.shuffle(tar_dataset)
        bool_shuffle = list_compare(src_dataset, tar_dataset)
    return src_dataset, tar_dataset


def create_dataset(train_dataset,
                   test_dataset,
                   batch_size):
    """ Create tf.data.Dataset.
    """
    # Create train dataset
    train_dataset_path = glob.glob(os.path.join(train_dataset + '*/*'))

    train_src_dataset = []
    train_tar_dataset = []
    for i in train_dataset_path:
        src_class, tar_class = interclassshuffle(i)
        for j in range(len(src_class)):
            train_src_dataset.append(src_class[j])
            train_tar_dataset.append(tar_class[j])

    train_src_dataset = tf.data.Dataset.from_tensor_slices(train_src_dataset)
    train_src_dataset = (
        train_src_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
    )

    train_tar_dataset = tf.data.Dataset.from_tensor_slices(train_tar_dataset)
    train_tar_dataset = (
        train_tar_dataset.map(load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
    )

    train_dataset = tf.data.Dataset.zip((train_src_dataset, train_tar_dataset))

    # Create test dataset
    test_dataset_path = glob.glob(os.path.join(test_dataset, '*/*'))

    test_src_dataset = []
    test_tar_dataset = []
    for i in test_dataset_path:
        src_class, tar_class = interclassshuffle(i)
        for j in range(len(src_class)):
            test_src_dataset.append(src_class[j])
            test_tar_dataset.append(tar_class[j])

    test_src_name_list = []
    for name in range(len(test_src_dataset)):
        test_src_name_list.append(test_src_dataset[name].split('\\')[3])

    test_src_dataset = tf.data.Dataset.from_tensor_slices(test_src_dataset)
    test_src_dataset = (
        test_src_dataset.map(test_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_tar_dataset = tf.data.Dataset.from_tensor_slices(test_tar_dataset)
    test_tar_dataset = (
        test_tar_dataset.map(test_load_image, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            .batch(batch_size, drop_remainder=True)
            .prefetch(tf.data.experimental.AUTOTUNE)
    )

    test_dataset = tf.data.Dataset.zip((test_src_dataset, test_tar_dataset))

    return train_dataset, test_dataset, test_src_name_list



if __name__ == '__main__':
    main(ArgParse())
