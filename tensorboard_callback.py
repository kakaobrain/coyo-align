import time

import tensorflow as tf
from tensorboard.compat.proto.event_pb2 import SessionLog
from tensorboard.compat.proto.event_pb2 import Event
from tensorflow.python.ops import summary_ops_v2


class CustomTensorBoard(tf.keras.callbacks.TensorBoard):
    def __init__(self, *args, **kwargs):
        super(CustomTensorBoard, self).__init__(*args, profile_batch=0, **kwargs)

    def on_train_begin(self, logs=None):
        super().on_train_begin(logs)

        # _train_step isn't restored from checkpoint. Restore it from optimizer state
        self._train_step.assign(self.model.optimizer.iterations)

        # Purge logs greater than _train_step. This deletes tensorboard logs written after the
        # checkpoint was saved. Since we're starting from the checkpoint, we discard logs that
        # we are going to reproduce.
        with self._train_writer.as_default():
            e = Event(step=self._train_step.numpy(), session_log=SessionLog(status=SessionLog.START))
            summary_ops_v2.import_event(tf.constant(e.SerializeToString(), dtype=tf.string))

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs.update({'epoch_time': time.time() - self.epoch_start})
        super().on_epoch_end(epoch, logs)

    def on_train_batch_end(self, batch, logs=None):
        if callable(self.model.optimizer.learning_rate):
            lr = self.model.optimizer.learning_rate(self.model.optimizer.iterations)
        else:
            lr = self.model.optimizer.learning_rate
        logs.update({'lr': lr})
        logs.update({'temperature': self.model.temperature})
        super().on_train_batch_end(batch, logs)