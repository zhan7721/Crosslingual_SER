<!-- GETTING STARTED -->
## Introduction

This is the codebase of "Cross-lingual Speech Emotion Recognition: Humans vs. Self-Supervised Models", including metadata, code of layer-wise analysis, PEFT fine-tuning, model results and logs. 

## Prerequisites

To use this repository: 
* clone the repository
  ```sh
  git clone https://github.com/zhan7721/Crosslingual_SER.git
  ```

* install requirements
  ```sh
  pip install requirements.txt
  ```

* prepare selected data (see split_id; the code uses privately stored data in HuggingFace)

## Train & test models

1. Train models:
  * To perform layer-wise analysis, run the following code with corresponding arguments:
    ```sh
    python runner_layer_wise_analysis.py --model cn --train cn --eval cn --test cn
    ```

  * To perform LoRA fine-tuning, run the following code with corresponding arguments:
    ```sh
    python runner_only_lora.py --model cn --train cn --eval cn --test cn --layer 6 --everylayer
    ```
  * To add various PEFT modules, or use two-stage fine-tuning, run the following code:
    ```sh
    python runner_various_peft.py --model cn --train cn --eval cn --test cn --layer 6 --everylayer --bottleneck --weightedgate --twostage
    ```
2. Test models on experimental speech data: 
  * Use test_on_exp_PEFT.py or de_mono_test_on_exp.py for the DE monolingual model:
    ```sh
    python test_on_exp_PEFT.py --model cn --condition ml
    ```

3. To see available arguments, please run:
  ```sh
  python some_script.py --help
  ```

## Contact
Zhichen Han - Z.Han-19@sms.ed.ac.uk

