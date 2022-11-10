import tensorflow as tf


def get_image_preprocess_fn(is_training):
    """
    Image preprocessor that matches ALIGN paper
    """
    def preprocess_fn(image):

        if is_training:
            image = tf.image.resize(image, (346, 346))
            image = tf.image.random_crop(image, (289, 289, 3))
            image = tf.image.random_flip_left_right(image)
        else:
            image = tf.image.resize(image, (346, 346))
            image = tf.image.crop_to_bounding_box(image, (346 - 289) // 2, (346 - 289) // 2, 289, 289)
        return image

    return preprocess_fn
