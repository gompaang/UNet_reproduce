# UNet_reproduce

The implementation of [[UNet]](https://arxiv.org/pdf/1505.04597.pdf). Dataset is [isbi-2012](https://github.com/alexklibisz/isbi-2012). I implemented it by referring to this [blog](https://89douner.tistory.com/298).

## Installation
- download the dataset.
  
  ```shell
  git clone https:github.com/gompaang/UNet_reproduce.git
  ```

- training

  ```shell
  CUDA_VISIBLE_DEVICES=0 python train.py
  ```
