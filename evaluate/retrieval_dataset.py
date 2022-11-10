import math
import tensorflow as tf
import tensorflow_datasets as tfds

from datasets.text_preprocessor import TextPreprocessor
from datasets.image_preprocessor import get_image_preprocess_fn


class RetrievalDataset:

    def __init__(self, dataset_path, global_batch_size, seq_length):
        self.dataset_path = dataset_path
        self.global_batch_size = global_batch_size
        self.text_preprocessor = TextPreprocessor(seq_length)

    def get_dataset(self):
        ds = tf.data.TFRecordDataset(self.dataset_path)

        feature = tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'image/id': tf.int64,
            'image/filename': tf.string,
            'captions': tfds.features.Sequence({'text': tf.string, 'id': tf.int64}),
        })
        return ds.map(feature.deserialize_example)

    def num_steps(self):
        return math.ceil(self.num_samples() / self.global_batch_size)

    @staticmethod
    def num_samples():
        return 1000

    def get_text_dataset(self):

        def parse_fn(f):
            # return image_id in shape of caption_id to match caption numbers
            idx = tf.repeat(f['image/id'], tf.shape(f['captions']['id']))
            text = f['captions']['text']
            return idx, text

        ds = self.get_dataset()
        ds = ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.unbatch()
        ds = ds.batch(batch_size=self.global_batch_size, drop_remainder=False)
        ds = ds.map(lambda idx, txt: (idx, self.text_preprocessor(txt)),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

    def get_image_dataset(self):
        image_preprocessor = get_image_preprocess_fn(is_training=False)

        def parse_fn(f):
            idx = f['image/id']
            image = f['image']
            return idx, image

        ds = self.get_dataset()
        ds = ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.map(lambda idx, img: (idx, image_preprocessor(img)),
                    num_parallel_calls=tf.data.experimental.AUTOTUNE)
        ds = ds.repeat(2) # TPU doesn't seem to handle remainder of last batch. Repeat and truncate it to be sure.
        ds = ds.batch(batch_size=self.global_batch_size, drop_remainder=False)
        ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

        return ds

class Flickr30kEvalDataset(RetrievalDataset):

    @staticmethod
    def num_samples():
        return 1000

class CocoEvalDataset(RetrievalDataset):

    @staticmethod
    def num_samples():
        return 5000
