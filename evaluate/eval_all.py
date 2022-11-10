import os
import sys
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import logging
import tensorflow as tf
from absl import app, flags

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from align import ALIGN
from evaluate.eval_knn import eval_knn
from evaluate.eval_retrieval import eval_retrieval
from datasets.text_preprocessor import TextPreprocessor
from retrieval_dataset import CocoEvalDataset, Flickr30kEvalDataset
from imagenet_dataset import ImageNetKNNEvalDataset

tf.get_logger().setLevel(logging.ERROR)
logging.getLogger("googleapiclient").setLevel(logging.WARNING)
logging.getLogger("oauth2client").setLevel(logging.WARNING)

flags.DEFINE_string('checkpoint', None, 'Tf checkpoint to load', required=True)
flags.DEFINE_string('workdir', None, 'Directory to save temporary files(e.g., tokenzier)', required=True)

flags.DEFINE_string('imagenet_dataset_dir', None, 'Directory containing imagenet2012 directory')
flags.DEFINE_string('coco_dataset_path', None, 'Path to coco.tfrecord')
flags.DEFINE_string('flickr_dataset_path', None, 'Path to flickr30k.tfrecord')


flags.DEFINE_string('img_encoder_name', 'efficientnet-b3', 'Image encoder name')
flags.DEFINE_string('txt_encoder_name', 'bert-mini', 'Text encoder name')
flags.DEFINE_integer('seq_length', 64, 'Maximum text tokens sequence length')
flags.DEFINE_integer('embed_dim', 640, 'Embedding dimension of image & text encoder')
flags.DEFINE_integer('batch_size', 1024, 'Global batch size')

flags.DEFINE_string('tpu', None, 'TPU name')

FLAGS = flags.FLAGS


def main(argv):
    del argv

    if FLAGS.tpu:
        resolver = tf.distribute.cluster_resolver.TPUClusterResolver(tpu=FLAGS.tpu)
        tf.config.experimental_connect_to_cluster(resolver)
        tf.tpu.experimental.initialize_tpu_system(resolver)
        strategy = tf.distribute.TPUStrategy(resolver)

        policy = tf.keras.mixed_precision.experimental.Policy('mixed_bfloat16')
        tf.keras.mixed_precision.experimental.set_policy(policy)
    else:
        strategy = tf.distribute.MirroredStrategy()

    os.environ["TFHUB_MODEL_LOAD_FORMAT"] = "UNCOMPRESSED"
    os.environ["TFHUB_CACHE_DIR"] = FLAGS.workdir

    with strategy.scope():
        text_prep = TextPreprocessor(FLAGS.seq_length)

        model = ALIGN(FLAGS.img_encoder_name, FLAGS.txt_encoder_name,
                      FLAGS.embed_dim, seq_length=FLAGS.seq_length, vocab_size=text_prep.get_vocab_size())
        model.compile()

        checkpoint = FLAGS.checkpoint
        print(f'Load checkpoint from [{checkpoint}]')
        model.load_weights(checkpoint)

        print(f'---------- {checkpoint} ----------')

        if FLAGS.imagenet_dataset_dir:
            dataset = ImageNetKNNEvalDataset(FLAGS.imagenet_dataset_dir, FLAGS.batch_size)
            knn_scores = eval_knn(dataset, model)
            print(f'ImageNet Knn: {knn_scores}')

        if FLAGS.coco_dataset_path:
            dataset = CocoEvalDataset(FLAGS.coco_dataset_path, FLAGS.batch_size, FLAGS.seq_length)
            retrieval_scores = eval_retrieval(dataset, model)
            print(f'MSCOCO I2T  : {retrieval_scores["I2T"]}')
            print(f'MSCOCO T2I  : {retrieval_scores["T2I"]}')

        if FLAGS.flickr_dataset_path:
            dataset = Flickr30kEvalDataset(FLAGS.flickr_dataset_path, FLAGS.batch_size, FLAGS.seq_length)
            retrieval_scores = eval_retrieval(dataset, model)
            print(f'Flickr30k I2T  : {retrieval_scores["I2T"]}')
            print(f'Flickr30k T2I  : {retrieval_scores["T2I"]}')

        print(f'---------- done ----------')


if __name__ == '__main__':
    app.run(main)
