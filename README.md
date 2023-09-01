# DeepSpeaker

Pytorch implemenation of SOTA Speaker Verification systems. 

## Installation

This repository has been created using python3.10. You can find the python3
dependencies on requirements.txt. Hence you can install it by:

* Clone the repo first as:
```bash
git clone https://github.com/miquelindia90/DeepSpeaker.git
```

* Create a virtual environment. We are using conda here but it could be done with others like virtualenv:
```bash
conda create -n deepspeaker python=3.10
conda activate deepspeaker
```

* Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

This repository shoud allow you to train a speaker embedding extractor according to the setup described in the paper. This speaker embedding extractor is based on a VGG-based classifier which identifies speaker identities given variable length audio utterances. The network used for this work uses log mel-spectogram features as input. Hence, we have added here the instructions to reproduce the feature extraction, the network training and the speaker embedding extraction step. Feel free to ask any doubt via git-hub issues, [twitter](https://twitter.com/mikiindia) or mail(miquelindia90@gmail.com).

### Network Training

First of all, It is needed to prepare some path files for the training step. The proposed models are trained as speaker classifiers, hence a classification-based loss and an accuracy metric will be used to monitorize the training progress. However in the validation step, an EER estimation is used to validate the network progress. The motivation behind this is that best accuracy models do not always have the best inter/intra speaker variability. Therefore we prefer to  use directly a task based metric to validate the model instead of using a classification one. Two different kind of path files will then be needed for the training/validation procedures:

Train Labels File (`train_labels_path`):

This file must have three columns separated by a blank space. The first column must contain the audio utterance paths, the second column must contain the speaker labels and the third one must be filled with -1. It is assumed that the labels correspond to the output network labels. Hence if you are working with a N speakers database, the speaker labels values should be in the 0 to N-1 range.

File Example:

<pre>
audiosPath/speaker1/audio1 0 -1
audiosPath/speaker1/audio2 0 -1
...
audiosPath/speakerN/audio4 N-1 -1</pre>

We have also added a `--train_data_dir` path argument. The dataloader will then look for the features in `--train_data_dir` + `audiosPath/speakeri/audioj` paths.

Valid Labels File:

For the validation step, it will be needed a tuple of client/impostors trial files. Client trials (`valid_clients`) file must contain pairs of audio utterances from same speakers and the impostors trials (`valid_impostors`) file must also contain audio utterance pairs but from different speakers. Each pair path must be separated with a blank space:

File Example (Clients):

<pre>
audiosPath/speaker1/audio1 audiosPath/speaker1/audio2
audiosPath/speaker1/audio1 audiosPath/speaker1/audio3

  
audiosPath/speakerN/audio4 audiosPath/speakerN/audio3</pre>

Similar to the train file, we have also added a `--valid_data_dir` argument.

Once you have all these data files ready, you can launch a model training with the following command:


```bash
python scripts/train.py config/config.yaml
```

With this script you will launch the model training with the default setup defined in `config/config.yaml`. The model will be trained following the methods and procedures described in the paper. The best models found will be saved in the `--out_dir` directory. You will find there a `.yaml` file with the training/model configuration and several checkpoint `.pt` files which store model weghts, optimizer state values, etc. The best saved models correspond to the last saved checkpoints.


### Network Embedding Extraction

(In construction)

## Support List:

* Topologies (SOTA Models)
    - [x] [VGG](https://arxiv.org/pdf/1906.09890.pdf)
    - [x] [ResNet34](https://arxiv.org/pdf/1512.03385.pdf)
    - [x] [ResNet101](https://arxiv.org/pdf/1512.03385.pdf)
    - [x] [RepVGG](https://arxiv.org/pdf/2101.03697.pdf) (Construction)
    - [x] [CAM++](https://arxiv.org/pdf/2303.00332.pdf) (Construction)
* Pooling Functions
    - [x] Temporal Average Pooling (TAP)
    - [x] Self Attentive Pooling (SAP)      
    - [x] Attentive Statistics Pooling (ASTP)
    - [x] [Self Multi-Attention Pooling (SMHA)](https://arxiv.org/pdf/1906.09890.pdf)
    - [x] [Double Multi-Attention Pooling (DMHA)](https://arxiv.org/pdf/1906.09890.pdf)
    - [x] [Multi-Query and Multi-Head Attentive Statistics Pooling (MQMHASTP)](https://arxiv.org/pdf/2110.05042.pdf) (Construction)
* Criteria
    - [x] Softmax
    - [x] [Add_Margin (AM-Softmax)](https://arxiv.org/pdf/1801.05599.pdf) (Construction)
    - [x] [Arc_Margin (AAM-Softmax)](https://arxiv.org/pdf/1801.07698v1.pdf)
    - [x] [Arc_Margin+Inter-topk+Sub-center](https://arxiv.org/pdf/2110.05042.pdf) (Construction)
* Scoring
    - [x] Cosine
    - [x] PLDA (Construction)
    - [x] Score Normalization (AS-Norm) (Construction)
* Metric
    - [x] EER
    - [x] minDCF (Construction)
* Online Augmentation
    - [x] Noise && RIR
    - [x] Speed Perturb
    - [x] SpecAug (Construction)
* Training Strategy
    - [x] Well-designed Learning Rate and Margin Schedulers
    - [x] Large Margin Fine-tuning
    - [x] Automatic Mixed Precision (AMP) Training
* Runtime 
    - [x] Python Binding (Construction)