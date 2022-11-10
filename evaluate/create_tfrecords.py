import os
import json

import tensorflow as tf
import tensorflow_datasets as tfds


def create_tfrecord(json_path, image_dir, tfrecord_filename):

    with open(json_path) as f:
        j = json.load(f)

    samples = [
        {'image': os.path.join(image_dir, sample['filename']),
         'image/id': sample['imgid'],
         'image/filename': sample['filename'],
         'captions': [{'id': caption['sentid'],
                       'text': caption['raw']}
                      for caption in sample['sentences']]
         }
        for sample in j['images'] if sample['split'] == 'test'
    ]

    print(f'number of samples in {tfrecord_filename} = {len(samples)}')

    with tf.io.TFRecordWriter(tfrecord_filename) as writer:

        feature = tfds.features.FeaturesDict({
            'image': tfds.features.Image(),
            'image/id': tf.int64,
            'image/filename': tf.string,
            'captions': tfds.features.Sequence({'text': tf.string, 'id': tf.int64}),
        })

        for sample in samples:
            writer.write(feature.serialize_example(sample))


if __name__ == '__main__':
    create_tfrecord("dataset_flickr30k.json",
                    "flickr30k/flickr30k-images",
                    "flickr30k.tfrecord")

    create_tfrecord("dataset_coco.json",
                    "coco/images/val2014",
                    "coco.tfrecord")
