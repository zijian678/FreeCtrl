# FreeCtrl
**FreeCtrl: Constructing Control Centers with Feedforward Layers for Learning-Free Controllable Text Generation (ACL 2024)** [Link](https://aclanthology.org/2024.acl-long.412/)


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
    title = "{F}ree{C}trl: Constructing Control Centers with Feedforward Layers for Learning-Free Controllable Text Generation",
    author = "Feng, Zijian  and
      Zhou, Hanzhang  and
      Mao, Kezhi  and
      Zhu, Zixiao",
    editor = "Ku, Lun-Wei  and
      Martins, Andre  and
      Srikumar, Vivek",
    booktitle = "Proceedings of the 62nd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = aug,
    year = "2024",
    address = "Bangkok, Thailand",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.acl-long.412",
    pages = "7627--7640",
    abstract = "Controllable text generation (CTG) seeks to craft texts adhering to specific attributes, traditionally employing learning-based techniques such as training, fine-tuning, or prefix-tuning with attribute-specific datasets. These approaches, while effective, demand extensive computational and data resources. In contrast, some proposed learning-free alternatives circumvent learning but often yield inferior results, exemplifying the fundamental machine learning trade-off between computational expense and model efficacy. To overcome these limitations, we propose FreeCtrl, a learning-free approach that dynamically adjusts the weights of selected feedforward neural network (FFN) vectors to steer the outputs of large language models (LLMs). FreeCtrl hinges on the principle that the weights of different FFN vectors influence the likelihood of different tokens appearing in the output. By identifying and adaptively adjusting the weights of attribute-related FFN vectors, FreeCtrl can control the output likelihood of attribute keywords in the generated content. Extensive experiments on single- and multi-attribute control reveal that the learning-free FreeCtrl outperforms other learning-free and learning-based methods, successfully resolving the dilemma between learning costs and model performance.",
}
```

## Contact
If you encounter any issues, please contact me at feng0119@e.ntu.edu.sg.
