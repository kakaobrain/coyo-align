import tensorflow as tf
import numpy as np

from align import ALIGN


def predict_text(self, features):
    idx, text = features
    return idx, self.text_encoder(text, training=False)

def predict_image(self, features):
    idx, image = features
    return idx, self.image_encoder(image, training=False)

def eval_retrieval(dataset, model):

    ALIGN.predict_step = predict_image
    model.compile()
    img_idxs, img_embs = model.predict(dataset.get_image_dataset(), verbose=1,
                                       steps=dataset.num_steps())

    img_idxs = img_idxs[:dataset.num_samples(),...]
    img_embs = img_embs[:dataset.num_samples(),...]

    ALIGN.predict_step = predict_text
    model.compile()
    txt_idxs, txt_embs = model.predict(dataset.get_text_dataset(), verbose=1)

    # np.save(f'eval_img_idxs.npy', img_idxs)
    # np.save(f'eval_img_embs.npy', img_embs)
    # np.save(f'eval_txt_idxs.npy', txt_idxs)
    # np.save(f'eval_txt_embs.npy', txt_embs)

    return retrieval_score(img_idxs, img_embs, txt_idxs, txt_embs)

def cosine_similarity(x, y):
    x /= tf.norm(x, ord=2, axis=-1, keepdims=True)
    y /= tf.norm(y, ord=2, axis=-1, keepdims=True)

    return tf.matmul(x, y, transpose_b=True)

def retrieval_score(img_idxs, img_embs, txt_idxs, txt_embs):

    cosine_sim = cosine_similarity(img_embs, txt_embs)

    def calc_recall_at_k(from_idxs, to_idxs, similiarity_mat):
        # transpose back and forth to top_k in axis=0 so that k shape can be broadcasted
        _, sorted_idx = tf.math.top_k(tf.transpose(similiarity_mat), k=10, sorted=True)
        sorted_idx = tf.transpose(sorted_idx).numpy()

        ret = {}
        for k in (1, 5, 10):
            top_k_match = (from_idxs == to_idxs[sorted_idx[:k,:]]).any(axis=0)
            recall_k = top_k_match.mean()
            ret[f'R@{k}'] = recall_k
        return ret

    # t2i
    t2i = calc_recall_at_k(txt_idxs, img_idxs, cosine_sim)
    t2i = {k:f'{v*100:.3f}' for k, v in t2i.items()}

    # i2t
    i2t = calc_recall_at_k(img_idxs, txt_idxs, tf.transpose(cosine_sim))
    i2t = {k:f'{v*100:.3f}' for k, v in i2t.items()}

    return {'I2T': i2t, 'T2I': t2i}


if __name__ == '__main__':
    img_idxs = np.load(f'eval_img_idxs.npy')
    img_embs = np.load(f'eval_img_embs.npy')
    txt_idxs = np.load(f'eval_txt_idxs.npy')
    txt_embs = np.load(f'eval_txt_embs.npy')
    retrieval_score(img_idxs, img_embs, txt_idxs, txt_embs)
