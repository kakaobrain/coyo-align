import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import logging as pylogging
import tensorflow_addons as tfa
import tensorflow as tf
from absl import app, flags, logging

tf.get_logger().setLevel(pylogging.WARNING)
pylogging.getLogger("googleapiclient").setLevel(pylogging.WARNING)
pylogging.getLogger("oauth2client").setLevel(pylogging.WARNING)

from align import ALIGN
from datasets.imgtxt_dataset import AlignDataset
from warmup_lr import extend_with_warmup_lr
from tensorboard_callback import CustomTensorBoard

flags.DEFINE_string('img_encoder_name', 'efficientnet-b3', 'Image encoder name')
flags.DEFINE_string('txt_encoder_name', 'bert-mini', 'Text encoder name')

flags.DEFINE_integer('batch_size', 2048, 'Global batch size')
flags.DEFINE_float('lr', 0.001, 'learning rate')
flags.DEFINE_float('weight_decay', 1e-5, 'Weight decay')
flags.DEFINE_integer('train_steps', 12000, 'Number of train steps')
flags.DEFINE_integer('warmup_steps', 100, 'Lr Warmup in steps')
flags.DEFINE_float('temperature', 1.0, 'init temperature')
flags.DEFINE_integer('embed_dim', 640, 'Embedding dimension of image & text encoder')
flags.DEFINE_integer('seq_length', 64, 'Maximum text tokens sequence length')

flags.DEFINE_string('dataset_dir', None, 'Directory containing .tfrecord files. Must be gs:// if TPU is used', required=True)
flags.DEFINE_string('outdir', None, 'Directory to save checkpoints and logs. Must be gs:// if TPU is used', required=True)

flags.DEFINE_string('tpu', None, 'TPU name')

FLAGS = flags.FLAGS


def main(argv) -> None:
    del argv

    tf.io.gfile.makedirs(FLAGS.outdir)

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
    os.environ["TFHUB_CACHE_DIR"] = FLAGS.outdir

    dataset = AlignDataset(FLAGS.dataset_dir, FLAGS.batch_size, FLAGS.seq_length)
    train_dataset = strategy.distribute_datasets_from_function(dataset.get_input_fn(is_training=True))

    with strategy.scope():
        model = ALIGN(FLAGS.img_encoder_name, FLAGS.txt_encoder_name,
                      embed_dim=FLAGS.embed_dim,
                      vocab_size=dataset.text_preprocessor.get_vocab_size(),
                      seq_length=FLAGS.seq_length,
                      temperature=FLAGS.temperature)

        warmed_up = extend_with_warmup_lr(tf.keras.optimizers.schedules.PolynomialDecay)
        lr_schedule = warmed_up(FLAGS.warmup_steps, FLAGS.lr, FLAGS.train_steps - FLAGS.warmup_steps,
                                end_learning_rate=0, power=1.0)
        optimizer = tfa.optimizers.LAMB(learning_rate=lr_schedule,
                                        weight_decay_rate=FLAGS.weight_decay,
                                        exclude_from_weight_decay=['bn/'])
        steps_per_execution = 100 # run this amount of steps in TPU wihtout coming back to CPU. It's faster to come back to CPU less often.
        model.compile(optimizer=optimizer, steps_per_execution=steps_per_execution,
                      run_eagerly=steps_per_execution == 1)

        latest = tf.train.latest_checkpoint(FLAGS.outdir)
        if latest:
            model.load_weights(latest)

            initial_epoch = int(os.path.basename(latest).split('_')[-1])
            logging.info(f'Training resume from {initial_epoch} epochs')
        else:
            initial_epoch = 0
            logging.info('Training started from scratch')

    callbacks = [
        CustomTensorBoard(log_dir=FLAGS.outdir, update_freq=steps_per_execution),
        tf.keras.callbacks.ModelCheckpoint(filepath=os.path.join(FLAGS.outdir, 'chpt_{epoch}'),
                                           save_weights_only=True),
        tf.keras.callbacks.ProgbarLogger(count_mode='steps',
                                         stateful_metrics={'loss', 'loss_i2t', 'loss_t2i', 'lr', 'temperature'}),
    ]

    # Training runs until train_steps. Epoch is used to manage how often checkpoint is saved.
    # Given the default 16k batch, we save checkpoint every 10_000 steps.
    # If we use larger batch, we save checkpoint proportionally more often.
    batch_scale = FLAGS.batch_size // (16 * 1024)
    steps_per_epoch = 10_000 // batch_scale if FLAGS.train_steps > 20_000 else 1_000
    logging.info(f'Training batch:{FLAGS.batch_size} batch_scale:{batch_scale} steps_per_epoch:{steps_per_epoch} epoch:{FLAGS.train_steps // steps_per_epoch}')

    assert FLAGS.train_steps % steps_per_epoch == 0
    model.fit(train_dataset,
              initial_epoch=initial_epoch,
              epochs=FLAGS.train_steps // steps_per_epoch,
              callbacks=callbacks,
              steps_per_epoch=steps_per_epoch)


if __name__ == '__main__':
    app.run(main)
