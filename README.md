# EEG classification with transformers

---

The GitHub repository for my thesis **"EEG classification with transformers"** on the IT University of Copenhagen, which investigates several different approaches for harnessing the power of transformer models for EEG classification.

##### Abstract

---

to be written

##### Usage

---

To run experiments first install the necessary packages:

```python
pip install -r requirements.txt
```

You should now be able to run experiments with:

```python
python scripts/run.py
```

If you have a [wandb](https://wandb.ai/) account, you can set 'wandb_logging' to 'True' in `scripts/support/configs.yml`under 'train_params'.

In the configs.yml file you can also specify which dataset, model and model hyperparameters to use.

**NB:** Not all dataset and model combinations work as some models expect raw EEG data as input and some expect the EEG data to have been transformed using permutation patterns as described in the thesis.

##### Dataset

---

I use the [BNCI competition IV 2b](https://www.bbci.de/competition/iv/) dataset obtained from [braindecode](https://braindecode.org/stable/index.html).
The dataset will automatically be downloaded the first time you run `scrpits/run.py`.
