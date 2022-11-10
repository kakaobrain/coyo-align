import math
import tensorflow as tf
import tensorflow_datasets as tfds
from datasets.image_preprocessor import get_image_preprocess_fn


class ImageNetKNNEvalDataset:

    def __init__(self, dataset_dir, global_batch_size):
        self.global_batch_size = global_batch_size

        self.builder = tfds.builder('imagenet2012:5.0.0', data_dir=dataset_dir)

    def num_samples(self, split):
        return self.builder.info.splits[split].num_examples

    def num_steps(self, split):
        return math.ceil(self.num_samples(split) / self.global_batch_size)

    def get_dataset(self, split):
        image_preprocessor = get_image_preprocess_fn(is_training=False)

        def parse_fn(f):
            return f['image'], f['label']

        ds = self.builder.as_dataset(split=split)
        ds = ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(lambda img, lab: (image_preprocessor(img), lab),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.repeat(2) # TPU doesn't seem to handle remainder of last batch. Repeat and truncate it to be sure.
        ds = ds.batch(batch_size=self.global_batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

