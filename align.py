import tensorflow as tf
from transformers import TFBertModel, BertConfig

from models import efficientnet

tf.get_logger().setLevel('ERROR')

def create_text_encoder(txt_encoder_name, embed_dim, seq_length, vocab_size):
    bert_inputs = dict(
        input_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        attention_mask = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
        token_type_ids = tf.keras.layers.Input(shape=(seq_length,), dtype=tf.int32),
    )
    map = {
        'bert-mini': (4, 256),
        'bert-base': (12, 768),
        'bert-large': (24, 1024),
    }
    l, h = map[txt_encoder_name.lower()]

    bert_config = BertConfig(vocab_size=vocab_size,
                             hidden_size=h, num_hidden_layers=l, num_attention_heads=h//64, intermediate_size=h*4)
    bert = TFBertModel(bert_config)

    embeddings = bert(bert_inputs, training=None)
    outputs = tf.keras.layers.Dense(embed_dim, dtype=tf.float32)(embeddings['last_hidden_state'][:,0,:])

    return tf.keras.Model(bert_inputs, outputs, name='text_encoder')


def create_vision_encoder(img_encoder_name, embed_dim):
    map = {
        'efficientnet-b0': efficientnet.EfficientNetB0,
        'efficientnet-b1': efficientnet.EfficientNetB1,
        'efficientnet-b2': efficientnet.EfficientNetB2,
        'efficientnet-b3': efficientnet.EfficientNetB3,
        'efficientnet-b4': efficientnet.EfficientNetB4,
        'efficientnet-b5': efficientnet.EfficientNetB5,
        'efficientnet-b6': efficientnet.EfficientNetB6,
        'efficientnet-b7': efficientnet.EfficientNetB7,
    }
    model = map[img_encoder_name.lower()](include_top=False, classifier_activation=None, pooling='avg', weights=None)
    model.trainable = True

    inputs = tf.keras.layers.Input(shape=(289, 289, 3), name="image_input")
    embeddings = model(inputs)
    if img_encoder_name == 'efficientnet-b7':
        outputs = embeddings
    else:
        outputs = tf.keras.layers.Dense(embed_dim, dtype=tf.float32)(embeddings)

    return tf.keras.Model(inputs, outputs, name='vision_encoder')


class ALIGN(tf.keras.models.Model):

    def get_config(self):
        config = super().get_config()
        config.update({
            'embed_dim': self.embed_dim,
            'seq_length': self.seq_length,
            'vocab_size': self.vocab_size,
        })
        return config

    def __init__(self,
                 img_encoder_name,
                 txt_encoder_name,
                 embed_dim,
                 seq_length,
                 vocab_size,
                 temperature=1.):
        super(ALIGN, self).__init__()
        self.embed_dim = embed_dim
        self.seq_length = seq_length
        self.vocab_size = vocab_size

        self.image_encoder = create_vision_encoder(img_encoder_name, embed_dim)
        self.text_encoder = create_text_encoder(txt_encoder_name, embed_dim, seq_length, vocab_size)

        self.temperature = tf.Variable(initial_value=temperature, trainable=True)

        self.loss_tracker = tf.keras.metrics.Mean(name="loss")
        self.loss_i2t_tracker = tf.keras.metrics.Mean(name="loss_i2t")
        self.loss_t2i_tracker = tf.keras.metrics.Mean(name="loss_t2i")

    def call(self, inputs, training):
        img, text = inputs

        image_features = self.image_encoder(img, training)
        text_features = self.text_encoder(text, training)

        return image_features, text_features

    def train_step(self, features):
        with tf.GradientTape() as tape:
            image_features, text_features = self(features, training=True)
            loss_image_to_text, loss_text_to_image = self.compute_loss(image_features, text_features)
            loss = loss_image_to_text + loss_text_to_image
            scaled_loss = tf.nn.compute_average_loss(loss)

        gradients = tape.gradient(scaled_loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))

        self.loss_tracker.update_state(loss)
        self.loss_i2t_tracker.update_state(loss_image_to_text)
        self.loss_t2i_tracker.update_state(loss_text_to_image)
        return {
            "loss": self.loss_tracker.result(),
            "loss_i2t": self.loss_i2t_tracker.result(),
            "loss_t2i": self.loss_t2i_tracker.result()
        }

    def predict_step(self, features):
        image_features, text_features = self(features, training=False)
        return image_features, text_features

    def compute_loss(self, image_features, text_features):
        image_features = tf.nn.l2_normalize(image_features, axis=-1)
        text_features = tf.nn.l2_normalize(text_features, axis=-1)

        replica_context = tf.distribute.get_replica_context()
        global_image_features = replica_context.all_gather(image_features, axis=0)
        global_text_features = replica_context.all_gather(text_features, axis=0)

        temperature = self.temperature

        logits_image_to_text = tf.matmul(image_features, global_text_features, transpose_b=True) / temperature
        logits_text_to_image = tf.matmul(text_features, global_image_features, transpose_b=True) / temperature

        batch_size = tf.shape(image_features)[0]
        global_batch_size = tf.shape(global_image_features)[0]

        replica_id = tf.cast(tf.cast(replica_context.replica_id_in_sync_group, tf.uint32), tf.int32)
        labels_idx = tf.range(batch_size) + replica_id * batch_size
        labels = tf.one_hot(labels_idx, global_batch_size)
        labels = tf.stop_gradient(labels)

        loss_image_to_text = tf.keras.losses.categorical_crossentropy(labels, logits_image_to_text, from_logits=True, label_smoothing=.1)
        loss_text_to_image = tf.keras.losses.categorical_crossentropy(labels, logits_text_to_image, from_logits=True, label_smoothing=.1)

        return loss_image_to_text, loss_text_to_image

    @property
    def metrics(self):
        # Let reset_metrics() work
        return [self.loss_tracker, self.loss_i2t_tracker, self.loss_t2i_tracker]
