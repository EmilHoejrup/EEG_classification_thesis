# EEG data encoding for classification with deep learning models

A GitHub repository for the thesis **"EEG data encoding for classification with deep learning models"** submitted at the IT University of Copenhagen. The thesis investigates several different approaches to harnessing the power of transformer models for EEG classification.

## Abstract

Being able to correctly classify electroencephalogram (EEG) data is of importance for diagnosing neural disorders such as Alzheimer's and Epilepsy. It is also an important research question in the field of Brain Computer Interface (BCI) applications, with the promise of helping impaired individuals control prosthetic devices. EEG is a non-invasive method for capturing brain activity with electrodes placed on the skull. Convolutional Neural Networks (CNNs) are the most popular deep learning architecture for decoding the EEG signal into a classification scheme that can be used by BCI devices. These models are good at capturing local features of the EEG signal but have no inherent way of capturing long term dependencies in the data. The transformer model architecture is at the core of the most popular Large Language Models (LLMs) such as GPT and BERT, but have also shown good performance in image classification tasks. The self-attention mechanism is good at capturing long term dependencies in a data sequence and therefore potentially a better alternative to CNN models for EEG classification. It is, however, not immediately clear how to transform EEG data into a representation that can be used effectively by a transformer model. In this thesis I investigate strategies for encoding EEG data into a representation that can effectively be used for EEG classification. The three major strategies I investigate are: 1. using the concept of permutation patterns from the field of permutation entropy to encode EEG data into "tokens" from a finite permutation pattern vocabulary, 2. combining a transformer model with a Graph Convolutional Network (GCN) module and 3. collapsing the temporal and spatial convolutions in the ShallowFBCSPNet and Conformer models into one spatiotemporal convolution. For the first two strategies I was not able to create well performing model implementations, but by collapsing the the temporal and spatial convolutions into a spatiotemporal convolution I find the ShallowFBCSPNet and Conformer models to perform at least as well or better than the original models with fewer parameters and a less complex model architecture.

## Usage

To run experiments first install the necessary packages:

```python
pip install -r requirements.txt
```

You should now be able to run experiments with:

```python
python scripts/run.py
```

If you have a [wandb](https://wandb.ai/) account, you can set 'wandb_logging' to 'True' in `scripts/support/configs.yml` under 'train_params'.

In the configs.yml file you can also specify which dataset, model and model hyperparameters to use.

**NB:** Not all combinations of datasets and models work. Some models expect raw EEG data as input and some expect the EEG data to have been transformed using permutation patterns as described in the thesis.

## Dataset

I use the [BNCI competition IV 2b](https://www.bbci.de/competition/iv/) dataset obtained from [braindecode](https://braindecode.org/stable/index.html).
The dataset will automatically be downloaded the first time you run `scripts/run.py`.
