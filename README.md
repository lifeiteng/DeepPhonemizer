# DeepPhonemizer

![Build Status](https://github.com/as-ideas/DeepPhonemizer/workflows/pytest/badge.svg)


DeepPhonemizer is a library for grapheme to phoneme conversion based on Transformer models. 
It is intended to be used in text-to-speech production systems with high accuracy and efficiency.
You can choose between a forward Transformer model (trained with CTC) and its autoregressive
counterpart. The former is faster and more stable whether the latter is slightly more accurate.

The main advantages of this repo are:

* Easy-to-use API for training and inference.
* Multilingual: You can train a single model on several languages.
* Accuracy: Phoneme and word error rates are comparable to state-of-art. 
* Speed: The repo is highly optimized for inference speed, phonemizing even large articles on a CPU is almost instantaneous if you choose the forward Transformer model.


Check out the [inference](https://colab.research.google.com/github/as-ideas/DeepPhonemizer/blob/main/dp/notebooks/Inference_Example.ipynb) and [training](https://colab.research.google.com/github/as-ideas/DeepPhonemizer/blob/main/dp/notebooks/Training_Example.ipynb) tutorial on colab! 



## Quickstart

Download the pretrained model: [en_us_cmudict_ipa](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa.pt)

```bash
from dp.phonemizer import Phonemizer

phonemizer = Phonemizer.from_checkpoint('en_us_cmudict_ipa.pt')
phonemizer('Phonemizing an English text is imposimpable!', lang='en_us')

'foʊnɪmaɪzɪŋ æn ɪŋglɪʃ tɛkst ɪz ɪmpəzɪmpəbəl!'
```


## Installation

```bash
pip install deep-phonemizer
```

### Training

All parameters are set in a config.yaml, you can find a config in the installed package under:
```bash
dp/configs/forward_config.yaml
```

Pepare data in a tuple-format and use the preprocess and train API:

```python
from dp.preprocess import preprocess
from dp.train import train

train_data = [
                ('en_us', 'young', 'jʌŋ'),
                ('de', 'benützten', 'bənʏt͡stn̩'),
                ('de', 'gewürz', 'ɡəvʏʁt͡s')
             ] * 100


preprocess(config_file='config.yaml', train_data=train_data)
train(config_file='config.yaml')
```
Model checkpoints will be stored in the checkpoints path that is provided by the config.yaml.

### Inference

Load the phonemizer from a checkpoint and run a prediction. By default, the phonemizer stores a 
dictionary of word-phoneme mappings that is applied first, and it uses the Transformer model
only to predict out-of-dictionary words.

```python
from dp.phonemizer import Phonemizer

phonemizer = Phonemizer.from_checkpoint('/content/checkpoints/best_model.pt')
result = phonemizer('Phonemizing an English text is imposimpable!', lang='en_us')
print(result)
```

If you need more inference information, you can use following API:

```python
from dp.phonemizer import Phonemizer
result = phonemizer.phonemise_list(['Phonemizing an English text is imposimpable!'], lang='en_us')

for word, pred in result.predictions.items():
  print(f'{word} {pred.phonemes} {pred.confidence}')
```


## Pretrained Models

| Model | Dataset | Commit |
|---|---|---|
|[foward_transformer](https://public-asai-dl-models.s3.eu-central-1.amazonaws.com/DeepPhonemizer/en_us_cmudict_ipa.pt) | [cmudict-ipa](https://github.com/menelik3/cmudict-ipa) | latest |


## Maintainers
* Christian Schäfer, github: [cschaefer26](https://github.com/cschaefer26)


## References

[Transformer based Grapheme-to-Phoneme Conversion](https://arxiv.org/abs/2004.06338)

[GRAPHEME-TO-PHONEME CONVERSION USING
LONG SHORT-TERM MEMORY RECURRENT NEURAL NETWORKS](https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/43264.pdf)
