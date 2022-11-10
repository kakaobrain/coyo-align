# COYO-ALIGN

COYO-ALIGN is an implementation of [ALIGN](https://arxiv.org/abs/2102.05918) by Kakao Brain that achieves similar performance to Google's ALIGN using the publicly available [COYO-700M](https://github.com/kakaobrain/coyo-dataset) dataset, instead of ALIGN 1.8B dataset which has not been released to the public.  When trained on the same dataset CC3M, COYO-ALIGN matches ALIGN performance.


|                                    | # of parameters |    Dataset    | ImageNet | Flickr30k |          |  MsCOCO  |          |
|------------------------------------|----------------:|:-------------:|:--------:|:---------:|:--------:|:--------:|:--------:|
|                                    |                 |               |   KNN    |  I2T R@1  | T2I R@1  | I2T R@1  | T2I R@1  |
| ALIGN-L2-Large(Google)             |       307M+117M |  ALIGN 1.8B   |   76.4   |   88.6    |   75.7   |   58.6   |   45.6   |
| ALIGN-B7-Base(Google)              |        66M+110M |  ALIGN 1.8B   |   69.3   |     -     |    -     |   55.4   |   41.7   |
| **COYO-ALIGN-B7-Base(KakaoBrain)** |        66M+110M | **COYO-700M** | **68.6** | **88.1**  | **73.2** | **61.2** | **43.1** |
|                                    |                 |               |          |           |          |          |          |
| ALIGN-B3-Mini(Google)              |       12M+11.3M |     CC-3M     |   48.9   |     -     |    -     |   22.1   |   17.3   |
| **COYO-ALIGN-B3-Mini(KakaoBrain)** |       12M+11.3M |     CC-3M     | **46.2** |   42.8    |   35.0   | **21.2** | **17.0** |

Note that only 86% of CC3M data is available for download as of Sept. 2022.

# Installation

    pip3 install -r requirements.txt

# Dataset

Datasets used

1. CC3M
2. [COYO-700M](https://github.com/kakaobrain/coyo-dataset)
3. ImageNet for ImageNet KNN evaluation
4. MS COCO Captions for I2T & T2I retrieval evaluation
5. Flickr30k for I2T & T2I retrieval evaluation

## CC-3M (174.6GiB)

Follow https://github.com/rom1504/img2dataset/blob/main/dataset_examples/cc3m.md to download CC3M. To be more specific:

1. Go to https://ai.google.com/research/ConceptualCaptions/download 
2. Download "Training split" as `Train_GCC-training.tsv`
3. 
    ```
   sed -i '1s/^/caption\turl\n/' Train_GCC-training.tsv
   
   pip install img2dataset tensorflow tensorflow_io
   # apt-get update && apt-get install -y libgl1
   img2dataset --url_list Train_GCC-training.tsv --input_format "tsv"\
               --url_col "url" --caption_col "caption" --output_format tfrecord\
               --output_folder cc3m --processes_count 32 --thread_count 64\
               --image_size 346 --resize_mode keep_ratio
    ```

## COYO-700M

Follow https://github.com/kakaobrain/coyo-dataset#getting-started to download images and save them into .tfrecord files

## ImageNet (for evaluation)

Follow https://www.tensorflow.org/datasets/catalog/imagenet2012

## MS COCO Captions & Flickr30k (for evaluation)

1. Download the followings:
   1. Validation set split configs: https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip
   2. MsCoco validation dataset: http://images.cocodataset.org/zips/val2014.zip
   3. Flickr30k dataset: https://www.kaggle.com/datasets/hsankesara/flickr-image-dataset
2. Unzip all
3. Run `python evaluate/create_tfrecords.py`

## Upload to GCS

TPU can only access filesystem on Google Cloud Storage. If you want to use TPU to train, follow the below steps to copy files to GCS.
```
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-402.0.0-linux-x86_64.tar.gz
tar -xf google-cloud-cli-402.0.0-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh
gcloud init
gsutil cp -R <tfrecord directory> gs://<your bucket>
```

# Train

## CC-3M + EfficientNet-B3 + BERT-mini setting

    python3 main.py --flagfile conf/cc3m.flags \
                    --outdir gs://<outdir> --dataset_dir gs://<directory containing tfrecord files> \
                    --tpu <tpu name>

## COYO-700M + EfficientNet-B7 + BERT-base setting

    python3 main.py --flagfile conf/coyo700.flags --flagfile conf/b7-base.flags \
                    --outdir gs://<outdir> --dataset_dir gs://<directory containing tfrecord files> \
                    --tpu <tpu name>

# Evaluate

## Download model weights

You can download the model weight from https://huggingface.co/kakaobrain/coyo-align-b7-base as follows:

    git lfs install
    git clone https://huggingface.co/kakaobrain/coyo-align-b7-base


## On TPU

Note that TPU can only access files on Google Cloud Storage. You have to upload the model weight files to a `gs://` location if you want to run evaluation on TPU.

    cd evaluate
    python eval_all.py --flagfile ../conf/b7-base.flags # if using b7 model
                       --checkpoint gs://<checkpoint path>
                       --workdir gs://<work dir> --tpu <tpu name>
                       --imagenet_dataset_dir gs://<directory containing imagenet2012>
                       --flickr_dataset_path gs://<path to flickr30k.tfrecord>
                       --coco_dataset_path gs://<path to coco.tfrecord>
   
If you want only to run ImageNet KNN evaluation, remove `--flickr_dataset_path` and `--coco_dataset_path` arguments

## On CPU

ImageNet KNN requires inferencing all 1.25M images in ImageNet. It takes very long time on CPU. It's recommended to use TPU or GPU for ImageNet KNN.
Flickr30k and Coco evaluation only evaluates on 5k and 1k images. So, it is doable on CPU. Remember to reduce `--batch_size` to fit within your machine's CPU memory. It takes around 30~60 minutes to finish.

    cd evaluate
    python eval_all.py --flagfile ../conf/b7-base.flags
                       --checkpoint coyo-align-b7-base/model-weights
                       --workdir ./ --batch_size 32
                       --flickr_dataset_path gs://<path to flickr30k.tfrecord>
                       --coco_dataset_path gs://<path to coco.tfrecord>

# Implementation note

Our implementation follows pretty much everything in the paper down to every detail. The only notable difference is use of `virtual_batch_size`. In our experiement on CC3M, the performance on V3-1024(as used by paper) was better than V3-128. We presumed it was because of the local batch size difference between V3-128 and V3-1024 since the same global batch size was used and the batch norm stats were not reduced across multi-nodes. `virtual_batch_size` was used to simulate the local batch size of V3-1024 on V3-128.(we preferred V3-128 because of its performance-to-price efficiency, especially on B3-Mini experiements in which the model size and batch size was small)

# Citation
```bibtex
@misc{kakaobrain2022coyo-align,
  title         = {COYO-ALIGN},
  author        = {Yoon, Boogeo and Lee, Youhan and Baek, Woonhyuk},
  year          = {2022},
  howpublished  = {\url{https://github.com/kakaobrain/coyo-align}},
}
```

# People
  - Boogeon Yoon ([@bgyoon](https://github.com/bgyoon))
  - Youhan Lee ([@qkqkfldis1](https://github.com/qkqkfldis1))
  - Woonhyuk Baek ([@wbaek](https://github.com/wbaek))


# Contact

[eric.yoon@kakaobrain.com](mailto:eric.yoon@kakaobrain.com)

# License

The source codes are licensed under Apache 2.0 License.