import numpy as np
import tensorflow as tf
from tqdm import tqdm

from align import ALIGN

@tf.function
def predict_step(self, data):
    image, label = data
    return self.image_encoder(image, training=False), label

def eval_knn(dataset, model):
    strategy = model.distribute_strategy

    train_dataset = dataset.get_dataset('train')
    len_train_dataset = dataset.num_steps('train')
    train_dataset = strategy.experimental_distribute_dataset(train_dataset)

    features, labels = [], []
    dataset_iterator = iter(train_dataset)

    ALIGN.predict_step = predict_step
    model.compile()

    for _ in tqdm(range(len_train_dataset)):
        feature, label = strategy.run(model.predict_step, args=(next(dataset_iterator),))

        features.append(strategy.gather(feature, axis=0))
        labels.append(strategy.gather(label, axis=0))

    train_img_embs = tf.concat(features, axis=0)
    train_labels = tf.concat(labels, axis=0)

    print(f'train_img_embs: {tf.shape(train_img_embs)}')
    print(f'train_labels: {tf.shape(train_labels)}')
    train_img_embs = train_img_embs[:dataset.num_samples('train'), ...]
    train_labels = train_labels[:dataset.num_samples('train'), ...]
    print(f'train_img_embs: {tf.shape(train_img_embs)}')
    print(f'train_labels: {tf.shape(train_labels)}')

    val_img_embs, val_labels = model.predict(dataset.get_dataset('validation'), verbose=1,
                                             steps=dataset.num_steps('validation'))

    print(f'val_img_embs: {tf.shape(val_img_embs)}')
    print(f'val_labels: {tf.shape(val_labels)}')
    val_img_embs = val_img_embs[:dataset.num_samples('validation'), ...]
    val_labels = val_labels[:dataset.num_samples('validation'), ...]
    print(f'val_img_embs: {tf.shape(val_img_embs)}')
    print(f'val_labels: {tf.shape(val_labels)}')

    return knn_score(strategy, train_img_embs, train_labels, val_img_embs, val_labels)

def index_lookup(indices, idx):
    rows = np.arange(idx.shape[0]).repeat(idx.shape[1])
    cols = idx.reshape(-1)
    return indices[rows, cols].reshape(idx.shape)

def knn_score(strategy, train_img_embs, train_labels, val_img_embs, val_labels):
    top_k = 1 # only works with top-1 on TPU

    @tf.function
    def top_k_similar(train_img_embs, val_img_embs):
        train_idxs, train_img_embs = train_img_embs

        train_img_embs /= tf.norm(train_img_embs, ord=2, axis=1, keepdims=True)
        val_img_embs /= tf.norm(val_img_embs, ord=2, axis=1, keepdims=True)

        similar = tf.matmul(val_img_embs, train_img_embs, transpose_b=True)
        values, indices = tf.math.top_k(similar, k=top_k)

        return values, tf.gather(train_idxs, indices)

    ds = tf.data.Dataset.from_tensor_slices(train_img_embs)
    ds = ds.enumerate()

    ds = ds.batch(8192, drop_remainder=False)
    ds = ds.prefetch(tf.data.experimental.AUTOTUNE)
    ds_len = len(ds)
    ds = strategy.experimental_distribute_dataset(ds)

    values, indices = [], []
    it = iter(ds)
    for _ in tqdm(range(ds_len)):
        v, i = strategy.run(top_k_similar, args=(next(it), val_img_embs))
        values.append(strategy.gather(v, axis=1))
        indices.append(strategy.gather(i, axis=1))

    values = tf.concat(values, axis=1)
    indices = tf.concat(indices, axis=1)

    _, idx = tf.math.top_k(values, k=top_k)
    indices = index_lookup(indices.numpy(), idx.numpy())
    train_labels = train_labels.numpy()

    scores = {}
    for k in range(top_k):
        top_k_match = (val_labels == train_labels[indices[:, :k+1]].T).any(axis=0)
        scores[f'R@{k+1}'] = f'{top_k_match.mean() * 100:.3f}'
    return scores
