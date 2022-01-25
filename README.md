<div align="center">
<img src="https://mmf.sh/img/logo.svg" width="50%"/>
</div>

#

<div align="center">
  <a href="https://mmf.sh/docs">
  <img alt="Documentation Status" src="https://readthedocs.org/projects/mmf/badge/?version=latest"/>
  </a>
  <a href="https://circleci.com/gh/facebookresearch/mmf">
  <img alt="CircleCI" src="https://circleci.com/gh/facebookresearch/mmf.svg?style=svg"/>
  </a>
</div>

---

MMF is a modular framework for vision and language multimodal research from Facebook AI Research. MMF contains reference implementations of state-of-the-art vision and language models and has powered multiple research projects at Facebook AI Research. See full list of project inside or built on MMF [here](https://mmf.sh/docs/notes/projects).

MMF is powered by PyTorch, allows distributed training and is un-opinionated, scalable and fast. Use MMF to **_bootstrap_** for your next vision and language multimodal research project by following the [installation instructions](https://mmf.sh/docs/). Take a look at list of MMF features [here](https://mmf.sh/docs/getting_started/features).

MMF also acts as **starter codebase** for challenges around vision and
language datasets (The Hateful Memes, TextVQA, TextCaps and VQA challenges). MMF was formerly known as Pythia. The next video shows an overview of how datasets and models work inside MMF. Checkout MMF's [video overview](https://mmf.sh/docs/getting_started/video_overview).

## Installation

Follow installation instructions in the [documentation](https://mmf.sh/docs/).

## Documentation

Learn more about MMF [here](https://mmf.sh/docs).

## Important Notes

This repository is forked from MMF frameworks original [repsoitory](https://github.com/facebookresearch/mmf).
For the purpose of this Reserach changes have been made into files with addresses given below:

- mmf/mmf/configs/models/visual_bert/defualts.yaml

- mmf/mmf/modules/embeddings.py

- mmf/mmf/models/visual_bert.py

- cuda version must be 10.2 for the code to work.

## Training 

For extracting base line follow the traning procedure as it is given below.

##### For pretraining coco model 

Run this command.

`mmf_run config=projects/pretrain_vl_right/configs/visual_bert/masked_coco/full.yaml run_type=train_val dataset=masked_coco model=visual_bert training.max_updates=30000 training.batch_size=64 training.find_unused_parameters=True`

##### For pretraining masked VQA model

After visual bert's pretraining on coco is completed, a final model will be saved in folder **"save"** in main mmf directory. Change this folders name to **"cocomodel"**, and then run the following command

`mmf_run config=projects/visual_bert/configs/masked_vqa2/pretrain_train_val.yaml run_type=train_val dataset=masked_vqa2 model=visual_bert training.max_updates=70000 training.batch_size=64 training.find_unused_parameters=True training.tensorboard=True checkpoint.resume_pretrained=True checkpoint.resume_file=./cocomodel/visual_bert_final.pth`

##### For fine-tuning the VQA model

After visual bert's pretraining on masked VQA model is completed, a final model will be saved in folder **"save"** in main mmf directory. Change this folders name to **"maskedvqamodel"**, and then run the following command

`mmf_run config=projects/visual_bert/configs/vqa2/train_val.yaml run_type=train_val dataset=vqa2 model=visual_bert training.max_updates=60000 training.batch_size=64 training.find_unused_parameters=True training.tensorboard=True checkpoint.resume_pretrained=True checkpoint.resume_file=./maskedvqamodel/visual_bert_final.pth`


## Testing

Run this command.

`mmf_predict config=projects/visual_bert/configs/vqa2/defaults.yaml model=visual_bert dataset=vqa2 run_type=test checkpoint.resume_file=./save/visual_bert_final.pth`

This will generate a JSON test file. This file has to be updated on EvalAI challenge [page](http://evalai.cloudcv.org/) to get accuracy on the dataset.

## For fine-tuning model on concatenated glove and bert embeddings

- Before training go to embeddings.py file on line 443 **"glove_embeddings code_block"** and uncomment it from **glove_embeddings code_block-start** to** glove_embeddings code_block-end**

- First make sure there is no folder name **"save"** in mmf main directory, if there is change its name to something else. Then follow the first two steps of the training process as it is to get maksed VQA model and then run the following command: 

`mmf_run config=projects/visual_bert/configs/vqa2/train_val.yaml run_type=train_val dataset=vqa2 model=visual_bert training.max_updates=60000 training.batch_size=64 training.find_unused_parameters=True training.tensorboard=True checkpoint.resume_pretrained=True checkpoint.resume_file=./maskedvqamodel/visual_bert_final.pth`


## Testing for model with concatenated embeddings

Run this command.

`mmf_predict config=projects/visual_bert/configs/vqa2/defaults.yaml model=visual_bert dataset=vqa2 run_type=test checkpoint.resume_file=./save/visual_bert_final.pth`

This will generate a JSON test file. This file has to be updated on EvalAI challenge [page](http://evalai.cloudcv.org/) to get accuracy on the dataset.




