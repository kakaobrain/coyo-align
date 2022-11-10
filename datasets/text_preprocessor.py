import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text # needed for bert_en_uncased_preprocess


class TextPreprocessor:
    def __init__(self, seq_length):
        super().__init__()
        self.preprocessor = hub.load("https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/3")
        self.model = self.create_text_preprocessor(seq_length)

    def create_text_preprocessor(self, seq_length):
        inputs = tf.keras.layers.Input(shape=(), dtype=tf.string)

        outputs = hub.KerasLayer(self.preprocessor.tokenize)(inputs)
        outputs = hub.KerasLayer(self.preprocessor.bert_pack_inputs,
                                 arguments=dict(seq_length=seq_length))([outputs])

        return tf.keras.Model(inputs, outputs)

    def get_vocab_size(self):
        return self.preprocessor.tokenize.get_special_tokens_dict()["vocab_size"]

    def __call__(self, text):
        text = self.model(text)

        # Map tf.hub preprocess output to huggingface BERT input
        text = {
            'input_ids': text['input_word_ids'],
            'attention_mask': text['input_mask'],
            'token_type_ids': text['input_type_ids'],
        }
        return text
