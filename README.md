# FreeCtrl
**FreeCtrl: Constructing Control Centers with Feedforward Layers for Learning-Free Controllable Text Generation (ACL 2024)** [Link](https://arxiv.org/abs/2406.09688)


## Text Generation

* **Download** and **unzip** the FFN distribution from [this link](https://drive.google.com/file/d/1QZIGzI7-f4AD1-r022QJ80E1Wo9vXoNr/view?usp=sharing).
* Navigate to the respective folders for text generation:
  * Topic Control
  * Sentiment Control
  * Multi-Attribute Control

Each folder contains the necessary files and instructions for generating text with different controls.


## Automatic Evaluation

For automatic evaluation, please use the classifiers trained by [PriorControl](https://github.com/HappyGu0524/MultiControl).

**Note:** There may be multiple acceptable names for the topic 'world', such as 'political' and 'government.' In our study, we use 'politics' and 'government'. Therefore, you may need to retrain this classifier to align with these terms. Other classifiers can remain unchanged.

## Acknowledgment
The code for FFN control is from [ff-layers](https://github.com/mega002/ff-layers/). The data and evaluation code is from [PriorControl](https://github.com/HappyGu0524/MultiControl). We thank the authors for their excellent contributions!

## Citation
If you find our work useful, please consider citing FreeCtrl:
```
@inproceedings{feng-etal-2024-freectrl,
    title = "FreeCtrl: Constructing Control Centers with Feedforward Layers for Learning-Free Controllable Text Generation",
    author = "Feng, Zijian  and
    Zhou Hanzhang and
    Zhu, Zixiao  and
    Mao, Kezhi",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    year = "2024",

}
```

## Contact
If you encounter any issues, please contact me at feng0119@e.ntu.edu.sg.
