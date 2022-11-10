import math
import tensorflow as tf

from datasets.text_preprocessor import TextPreprocessor
from datasets.image_preprocessor import get_image_preprocess_fn


class AlignDataset:
    def __init__(self, dataset_dir, global_batch_size, seq_length):
        self.global_batch_size = global_batch_size

        self.tfrecord_files = tf.io.gfile.glob(f'{dataset_dir}/*.tfrecord')

        self.text_preprocessor = TextPreprocessor(seq_length)

    def get_input_fn(self, is_training):
        image_preprocessor = get_image_preprocess_fn(is_training)

        def parse_fn(record):
            f = tf.io.parse_example(record, {
                'jpg': tf.io.FixedLenFeature([], tf.string),
                'txt': tf.io.FixedLenFeature([], tf.string),
            })

            return tf.io.decode_image(f['jpg'], channels=3, expand_animations=False), f['txt']

        def input_fn(input_context):
            local_batch_size = input_context.get_per_replica_batch_size(self.global_batch_size)

            ds = tf.data.Dataset.from_tensor_slices(self.tfrecord_files)
            ds = ds.shard(input_context.num_input_pipelines, input_context.input_pipeline_id)
            if is_training:
                # For perfect shuffling, a buffer size greater than or equal to the full size of the dataset is required
                ds = ds.shuffle(math.ceil(len(self.tfrecord_files) / input_context.num_input_pipelines))

            ds = ds.interleave(tf.data.TFRecordDataset, num_parallel_calls=tf.data.experimental.AUTOTUNE, deterministic=False)

            if is_training:
                ds = ds.shuffle(16 * local_batch_size)
            ds = ds.repeat()

            ds = ds.map(parse_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.map(lambda img, txt: (image_preprocessor(img), txt), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.batch(batch_size=local_batch_size, drop_remainder=is_training)
            ds = ds.map(lambda img, txt: (img, self.text_preprocessor(txt)), num_parallel_calls=tf.data.experimental.AUTOTUNE)
            ds = ds.prefetch(tf.data.experimental.AUTOTUNE)

            # from deepmind's code : https://github.com/deepmind/deepmind-research/blob/master/byol/utils/dataset.py#L91
            options = tf.data.Options()
            options.experimental_deterministic = False
            options.experimental_threading.private_threadpool_size = 48
            options.experimental_threading.max_intra_op_parallelism = 1
            ds = ds.with_options(options)

            return ds

        return input_fn

if __name__ == '__main__':
    tfrecord_files = tf.io.gfile.glob('cc3m/*.tfrecord')
    ds = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    ds = ds.interleave(tf.data.TFRecordDataset)
    for record in ds.take(1):
        example = tf.train.Example()
        example.ParseFromString(record.numpy())
        print([k for k in example.features.feature.keys()])