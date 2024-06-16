# Single Topic Control

## Text Generation

```
python -u single_topic_control.py
```

or
```
bash run_experiment.sh
```

## Hyperparameters

**For reproduction of our results or comparison to your method, please use grid search to obtain best results.**

Parameters and Ranges

--sentence_thre 1.0 - 1.2

--lamda 0.1 - 2.0

--k_values 10 - 100

--label_word_file the attribute keywords can be changed in this file to generate sentences containing different keywords

--num_beams, --top_k, --temperature can be adjusted to improve the **diversity** and **fluency** of generated sentences.

## Output
The generated sentences are saved to the folder of output.
