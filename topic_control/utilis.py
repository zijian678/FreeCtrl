from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch.nn.functional as F
# from step1_cluster_selection import args
import sys
import types

import math
from collections import Counter

def load_model(device,model_name):
    """
    """
    print('Base LM:', model_name, '\n')

    # Load Base (Pre-Trained) Tokenizer, LM
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    pretrained = AutoModelForCausalLM.from_pretrained(model_name).to(device)

    return tokenizer, pretrained

def rank_all_projected_values(model,target):
    # logits = []
    all_scores = []
    all_values = []
    for i in tqdm(range(model.config.n_layer)):
        # print('layer:',i)
        layer_logits = torch.matmul(model.transformer.wte.weight, model.transformer.h[i].mlp.c_proj.weight.T).T
        # print('layer_logits:',layer_logits.shape,layer_logits) # layer_logits: torch.Size([4096, 50257])
        # logits.append(layer_logits)
        layer_logits = F.softmax(layer_logits,dim = -1)
        target_logits = layer_logits[:,target] # target_ids: torch.Size([4096, 376])
        # print('target_logits:',target_logits.shape)
        # target_sum = torch.sum(target_logits,dim = -1)
        target_sum, _ = torch.max(target_logits, dim=1)
        # print('target_sum:',target_sum.shape,target_sum) # torch.Size([4096])
        for idx in range(len(target_sum)):
            all_scores.append(float(target_sum[idx]))
            all_values.append((i,idx))
    return all_scores, all_values


def organize_list(input_list):
    organized_dict = {}
    for key, value in input_list:
        if key in organized_dict:
            organized_dict[key].append(value)
        else:
            organized_dict[key] = [value]

    # Sort the lists for each key
    for key in organized_dict:
        organized_dict[key].sort()

    # Sort the dictionary by keys
    sorted_dict = {k: organized_dict[k] for k in sorted(organized_dict)}

    return sorted_dict

def distinctness2(generations):
    dist1, dist2, dist3 = [], [], []
    unigrams, bigrams, trigrams = set(), set(), set()
    total_words = 0
    for gen in generations:
        o = gen.split(' ')
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
    dist1.append(len(unigrams) / total_words)
    dist2.append(len(bigrams) / total_words)
    dist3.append(len(trigrams) / total_words)

    return np.nanmean(dist1), np.nanmean(dist2), np.nanmean(dist3)

def eva_dis(generations):
    dist1, dist2, dist3 = [], [], []
    unigrams, bigrams, trigrams = set(), set(), set()
    total_words = 0
    for gen in generations:
        o = gen.split(' ')
        total_words += len(o)
        unigrams.update(o)
        for i in range(len(o) - 1):
            bigrams.add(o[i] + '_' + o[i + 1])
        for i in range(len(o) - 2):
            trigrams.add(o[i] + '_' + o[i + 1] + '_' + o[i + 2])
    dist1.append(len(unigrams) / total_words)
    dist2.append(len(bigrams) / total_words)
    dist3.append(len(trigrams) / total_words)
    # calculate dist1, dist2, dist3 across generations for every prompt
    # for i, row in tqdm(generations_df.iterrows(), total=len(generations_df.index), desc='Evaluating diversity'):
    #     generations = [g['text'] for g in row['generations']]
    #     # print('generations:',generations)
    #     generations = generations[:1]
        # print('generations:', len(generations),generations)


    # take the mean across prompts
    final_score = (np.nanmean(dist1) + np.nanmean(dist2) + np.nanmean(dist3))/3.0
    return final_score
def average_diversity_score(cluster):
    # Calculate the average diversity score for a cluster
    return eva_dis([" ".join(sentences) for sentences in cluster])

def modified_silhouette_coefficient(clusters):
    silhouette_scores = []
    cluster_diversity_scores = [average_diversity_score(cluster) for cluster in clusters]
    print('cluster_diversity_scores:',cluster_diversity_scores)

    for i, cluster in enumerate(clusters):

        a = cluster_diversity_scores[i]

        # Separation: Diversity to the nearest cluster
        # nearest_cluster_diversity = min(
        #     [score for j, score in enumerate(cluster_diversity_scores) if j != i]
        # )
        nearest_cluster_diversity = min(
            [average_diversity_score(cluster + anc) for j, anc in enumerate(clusters) if j != i]
        )
        # print('nearest_cluster_diversity:', nearest_cluster_diversity)

        b = nearest_cluster_diversity * 2

        # Silhouette score for the sentence
        silhouette_score = (b - a) / max(a, b)
        silhouette_scores.append(silhouette_score)



    # Average silhouette score for all sentences
    return sum(silhouette_scores) / len(silhouette_scores)

from typing import List, Optional, Tuple, Dict, Union
from torch.nn import CrossEntropyLoss
import torch
from transformers import GPT2Tokenizer, GPT2LMHeadModel

import numpy as np
import torch.nn.functional as F


class GPT2Wrapper():
    def __init__(self, model_name: str = "gpt2-medium", use_cuda: bool = False):
        """
        :param model_name: the name of the pretrained GPT2 model (default: "gpt2-medium")
        :param use_cuda: whether to use CUDA
        """
        self._device = "cuda:0" if torch.cuda.is_available() and use_cuda else "cpu"

        self._tokenizer = GPT2Tokenizer.from_pretrained(model_name)
        self._model = GPT2LMHeadModel.from_pretrained(model_name)
        if use_cuda:
            self._model.parallelize()
        self._tokenizer.pad_token = self._tokenizer.eos_token
        self._model.config.pad_token_id = self._tokenizer.eos_token_id

        self.hooks = None

    def query_model_tok_dist(self, prompt):
        tokens = self._tokenizer.encode_plus(prompt, return_tensors='pt').to(self._device)
        output = self._model(**tokens)
        logits = output['logits']
        probs = F.softmax(logits[0][tokens['input_ids'].shape[1] - 1], dim=-1)  # gets probs after last tok in seq

        probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

        # assert probs add to 1
        assert np.abs(np.sum(probs) - 1) <= 0.01, str(np.abs(np.sum(probs) - 1))

        probs_ = []
        for index, prob in enumerate(probs):
            probs_.append((index, prob))

        top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:10]
        top_k = [(t[1].item(), self._tokenizer.decode(t[0])) for t in top_k]

        return top_k
    def query_logits(self, prompt):
        tokens = self._tokenizer.encode_plus(prompt, return_tensors='pt').to(self._device)
        output = self._model(**tokens)
        logits = output['logits']
        final_logits = logits[0][tokens['input_ids'].shape[1] - 1]
        final_logits = torch.reshape(final_logits, (-1,)).detach().cpu()

        ### output ptobs
        # probs = F.softmax(logits[0][tokens['input_ids'].shape[1] - 1], dim=-1)  # gets probs after last tok in seq
        #
        # probs = torch.reshape(probs, (-1,)).detach().cpu()# .numpy()

        # assert probs add to

        return  final_logits

    def query_all_logits(self, prompt):
        tokens = self._tokenizer.encode_plus(prompt, return_tensors='pt').to(self._device)
        output = self._model(**tokens)
        logits = output['logits']
        final_logits = logits[0]
        final_logits = final_logits.detach().cpu()
        # final_logits = torch.reshape(final_logits, (-1,)).detach().cpu()

        ### output ptobs
        # probs = F.softmax(logits[0][tokens['input_ids'].shape[1] - 1], dim=-1)  # gets probs after last tok in seq
        #
        # probs = torch.reshape(probs, (-1,)).detach().cpu()# .numpy()

        # assert probs add to

        return  final_logits

    def generate(self, input_text: List[str], word_filter: bool = False, min_length: int = 20, max_length: int = 20,
                 **kwargs):
        inputs = self._tokenizer.batch_encode_plus(input_text, padding=True, return_tensors='pt')
        inputs['attention_mask'] = torch.flip(inputs['attention_mask'], dims=[1])
        shifts = inputs['attention_mask'].shape[-1] - inputs['attention_mask'].sum(dim=-1)
        for batch_idx in range(inputs['input_ids'].shape[0]):
            inputs['input_ids'][batch_idx] = inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_length = inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length

        # print('input ids:', inputs['input_ids'])
        # print('input ids:',self._tokenizer.convert_ids_to_tokens(inputs['input_ids']))

        output_ids = self._model.generate(**inputs, min_length=min_length, max_length=max_length, **kwargs)

        # only return the continuation text
        batch_size = output_ids.shape[0]
        output_ids = output_ids[:batch_size, inputs['input_ids'].shape[1]:]

        return self._tokenizer.batch_decode(output_ids)

    def project_value_to_vocab(self, layer, value_idx, top_k=10):
        normed = self._model.transformer.ln_f(self._model.transformer.h[layer].mlp.c_proj.weight.data[value_idx]).to(
            self._device)

        logits = torch.matmul(self._model.lm_head.weight, normed.T).to(self._device)
        probs = F.softmax(logits, dim=-1)
        probs = torch.reshape(probs, (-1,)).detach().cpu().numpy()

        probs_ = []
        for index, prob in enumerate(probs):
            probs_.append((index, prob))

        top_k = sorted(probs_, key=lambda x: x[1], reverse=True)[:top_k]
        value_preds = [(self._tokenizer.decode(t[0]), t[1]) for t in top_k]

        return value_preds

    def generate_word_filter(self,
                             prompt: Union[str, List[str]],
                             bad_words: List[str],
                             min_length: int = 20,
                             max_length: int = 20,

                             **model_kwargs) -> List[str]:

        # bad_words = open("word_filter_words.txt").read().split("\n")
        bad_words_ids = [
            self._tokenizer.encode(bad_word, add_prefix_space=True)
            for bad_word in bad_words
        ]
        # print('bad_words_ids:',bad_words_ids)

        if isinstance(prompt, str):
            prompt = [prompt]

        inputs = self._tokenizer.batch_encode_plus(prompt, padding=True, return_tensors='pt')
        inputs['attention_mask'] = torch.flip(inputs['attention_mask'], dims=[1])
        shifts = inputs['attention_mask'].shape[-1] - inputs['attention_mask'].sum(dim=-1)
        for batch_idx in range(inputs['input_ids'].shape[0]):
            inputs['input_ids'][batch_idx] = inputs['input_ids'][batch_idx].roll(shifts[batch_idx].item())

        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_length = inputs['input_ids'].shape[1]
        if min_length is not None:
            min_length = min_length + input_length
        if max_length is not None:
            max_length = max_length + input_length

        output_ids = self._model.generate(**inputs,
                                          min_length=min_length, max_length=max_length,
                                          bad_words_ids=bad_words_ids,
                                          **model_kwargs)

        # only return the continuation text
        batch_size = output_ids.shape[0]
        output_ids = output_ids[:batch_size, inputs['input_ids'].shape[1]:]

        return self._tokenizer.batch_decode(output_ids)

    def compute_loss(self, input_ids: torch.LongTensor, labels: torch.LongTensor) -> torch.Tensor:
        outputs = self._model(input_ids, labels=labels)
        lm_logits = outputs[1]

        # Shift so that tokens < n predict n
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        # Flatten the tokens
        loss_fct = CrossEntropyLoss()
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        return loss

    def set_value_activations(self,
                              values_per_layer: Dict[int, List[int]],
                              coef_value: int = 3):
        """
        Uses PyTorch hooks to set the activations of each value in values_per_layer to coef_value
        Only works on GPT2 from HF.

        :params values_per_layer: dictionary with keys that correspond to layers and values that correspond to indices
        :params coef_value: number to set the values' activations to
        """

        def value_activation_replacement_hook(values, coef_val):
            def hook(module, input, output):
                output[:, :, values] = coef_val

            return hook

        hooks = []

        for layer in range(self._model.config.n_layer):
            if layer in values_per_layer:
                values = values_per_layer[layer]
            else:
                values = []

            hook = self._model.transformer.h[layer].mlp.c_fc.register_forward_hook(
                value_activation_replacement_hook(values, coef_value)
            )

            hooks.append(hook)

        self.hooks = hooks

        return hooks

    def remove_all_hooks(self):
        if self.hooks is not None:
            for hook in self.hooks:
                hook.remove()

            self.hooks = None
        else:
            print("No hooks to remove")

def get_all_value_vectors(model):
    vec = []
    for i in tqdm(range(model.config.n_layer)):
        current_v = model.transformer.h[i].mlp.c_proj.weight
        vec.append(current_v)


    # vec = torch.cat(vec,dim = 0)
    return vec

def vasualiza_proj(model,tokenizer,layer_id,vec_id):
    layer_logits = torch.matmul(model.transformer.wte.weight, model.transformer.h[layer_id].mlp.c_proj.weight.T).T
    target_logits = layer_logits[vec_id]
    target_logits = F.softmax(target_logits,dim = -1)
    kvals,kinds = torch.topk(target_logits,k = 30)
    print('topk values:',kvals)
    print('topk tokens:',tokenizer.convert_ids_to_tokens(kinds))

def get_dominant_repre(model):
    vec = []
    emb_weight = model.transformer.wte.weight
    for i in tqdm(range(model.config.n_layer)):
        current_v = model.transformer.h[i].mlp.c_proj.weight
        c_proj = current_v @ emb_weight.t() # torch.Size([4096, 50257])

        embs = []
        for xx in c_proj:
            kv,kind = torch.topk(xx,k=1)
            cemb = emb_weight[kind[0]]
            embs.append(cemb.unsqueeze(0))

        # print('c_proj:',c_proj.shape)
        embs = torch.cat(embs,dim = 0)# embs: torch.Size([4096, 1024])
        # print('embs:',embs.shape)

        vec.append(embs.unsqueeze(0))

    # vec = torch.cat(vec,dim = 0)
    return vec

def get_k_dominant_repre(model,k):
    vec = []
    emb_weight = model.transformer.wte.weight
    emb_weight = emb_weight.detach().cpu()
    for i in tqdm(range(model.config.n_layer)):
        current_v = model.transformer.h[i].mlp.c_proj.weight
        current_v = current_v.detach().cpu()
        c_proj = current_v @ emb_weight.t() # torch.Size([4096, 50257])
        c_proj = F.softmax(c_proj,dim = -1)

        embs = []
        for xx in c_proj:
            kv,kind = torch.topk(xx,k=k)
            cemb = emb_weight[kind] # cemb: torch.Size([5, 1024])
            # cemb = cemb
            # print('kv:',kv)
            # print('cemb:',cemb.shape)
            weights = F.normalize(kv,p=1,dim = -1)
            weights = weights.unsqueeze(0)
            # print('weights:',weights)
            final_repre = weights @ cemb
            # final_repre = torch.mean(final_repre)
            # print('final_repre:',final_repre.shape)
            embs.append(final_repre)

        # print('c_proj:',c_proj.shape)
        embs = torch.cat(embs,dim = 0)# embs: torch.Size([4096, 1024])
        # print('embs:',embs.shape)

        vec.append(embs.unsqueeze(0))

    # vec = torch.cat(vec,dim = 0)
    return vec

def get_v_repre(model):
    vec = []
    emb_weight = model.transformer.wte.weight
    for i in tqdm(range(model.config.n_layer)):
        current_v = model.transformer.h[i].mlp.c_proj.weight

        vec.append(current_v.unsqueeze(0))

    # vec = torch.cat(vec,dim = 0)
    return vec

def organize_list(input_list):
    organized_dict = {}
    for key, value in input_list:
        if key in organized_dict:
            organized_dict[key].append(value)
        else:
            organized_dict[key] = [value]

    # Sort the lists for each key
    for key in organized_dict:
        organized_dict[key].sort()

    # Sort the dictionary by keys
    sorted_dict = {k: organized_dict[k] for k in sorted(organized_dict)}

    return sorted_dict

def obtain_contrastive_prob(logits,label_ids):

    all_scores = []
    for kk in label_ids:
        cids = label_ids[kk]
        current_diff = logits[:, cids]  # torch.Size([98304, 143])
        # print('cids:',len(cids),'current_diff:',current_diff.shape)
        current_diff = torch.mean(current_diff, dim=-1)  # torch.Size([98304])
        # print('current_diff:',current_diff.shape)
        all_scores.append(current_diff.unsqueeze(1))

    all_scores = torch.cat(all_scores, dim=1)
    # print('all_scores:',all_scores.shape,all_scores)
    all_scores = F.softmax(all_scores,dim = -1)
    return all_scores

def obtain_overall_prob(logits,label_ids):
    logits = F.softmax(logits, dim = -1)

    all_scores = []
    for kk in label_ids:
        cids = label_ids[kk]
        current_diff = logits[:, cids]  # torch.Size([98304, 143])
        # print('cids:',len(cids),'current_diff:',current_diff.shape)
        current_diff = torch.sum(current_diff, dim=-1)  # torch.Size([98304])
        # print('current_diff:',current_diff.shape)
        all_scores.append(current_diff.unsqueeze(1))

    all_scores = torch.cat(all_scores, dim=1)
    # all_scores = F.softmax(all_scores,dim = -1)
    return all_scores

def get_control_probs(v_positions,wrapper):
    control_probs = []
    coef_value = 50
    prompt = ' '
    # vcs = [3.0,6.0,10.0,20.0,50.0,100,200]
    for i in tqdm(range(len(v_positions))):
        value_pos = [v_positions[i]]
        value_dict = organize_list(value_pos)
        # print('value_dict:',value_dict)

        wrapper.set_value_activations(value_dict, coef_value=coef_value)
        new_all_logits = wrapper.query_logits(prompt)
        # print('new_all_logits:',new_all_logits.shape,new_all_logits)
        new_all_logits = F.softmax(new_all_logits, dim=-1)
        wrapper.remove_all_hooks()

    # observe:
    # kv,kids = torch.topk(new_all_logits,k=10)
    # print('kv:',kv)
    # print('kids:',kids,tokenizer.convert_ids_to_tokens(kids))
    # new_all_logits = torch.mean(new_all_logits,dim = 0)
    # print('final new_all_logits:',new_all_logits.shape,new_all_logits)
    control_probs.append(new_all_logits.unsqueeze(0))

    control_probs = torch.cat(control_probs, dim=0)

    return control_probs

def de_noising(class_repre,vec,label_id = 0,label_num = 4):
    sim = F.cosine_similarity(class_repre, vec.unsqueeze(0), dim=1).squeeze()
    # print('sim:',sim)

    s = sim[label_id] * (label_num - 1)/(torch.sum(sim) - sim[label_id])
    return s

def scoring(all_label_repre,sen_emb,label_id = 0,label_num = 4):
    # print('sen_emb:',sen_emb.shape)

    sim = []
    for lr in all_label_repre:
        csim = F.cosine_similarity(lr.unsqueeze(1), sen_emb.unsqueeze(0), dim=2)
        max_values, _ = torch.max(csim, dim=1)

        kvs,kinds = torch.topk(max_values,k = len(max_values))
        # print('csim:', csim.shape, 'max_values:', max_values.shape,'kvs:',kvs)
        ss = torch.mean(kvs)
        sim.append(float(ss))
    # sim = F.cosine_similarity(class_repre, vec.unsqueeze(0), dim=1).squeeze()
    # print('sim:',sim)
    sim = torch.tensor(sim)


    s = sim[label_id] * (label_num - 1)/(torch.sum(sim) - sim[label_id])
    # print('final sim:', sim,'final score:',s)
    return s

# def scoring(all_label_repre,sen_emb,label_id = 0,label_num = 4):
#     # print('sen_emb:',sen_emb.shape)
#
#     sim = []
#     for lr in all_label_repre:
#         csim = F.cosine_similarity(lr.unsqueeze(1), sen_emb.unsqueeze(0), dim=2)
#         # print('csim:',csim.shape)
#         # max_values, _ = torch.max(csim, dim=1)
#         #
#         # kvs,kinds = torch.topk(max_values,k = 20)
#         # print('csim:', csim.shape, 'max_values:', max_values.shape,'kvs:',kvs)
#         ss = torch.mean(csim[0])
#         sim.append(float(ss))
#     # sim = F.cosine_similarity(class_repre, vec.unsqueeze(0), dim=1).squeeze()
#     # print('sim:',sim)
#     sim = torch.tensor(sim)
#
#
#     s = sim[label_id] * (label_num - 1)/(torch.sum(sim) - sim[label_id])
#     # print('final sim:', sim,'final score:',s)
#     return s

def calculate_entropy(word_counts):
    # print('word_counts:',word_counts)
    total_words = sum(word_counts.values())
    # Filter out zero-count words to avoid math domain error
    word_probabilities = {word: count / total_words for word, count in word_counts.items() if count > 0}
    # Calculate entropy, excluding zero-probability words
    # print('word_probabilities:',word_probabilities)
    return -sum(prob * math.log(prob, 2) for prob in word_probabilities.values())


def update_ban(all_words,threshold_entropy):
    word_counts = Counter(all_words)

    # Initialize banned_list
    banned_list = []

    # Hyperparameter: Entropy threshold
    # threshold_entropy = 1.5  # Adjust this threshold based on requirements

    # Function to adjust the banned list
    current_entropy = calculate_entropy(word_counts)
    words_sorted_by_freq = sorted(word_counts, key=word_counts.get, reverse=True)
    # print('original current_entropy:',current_entropy)
    # print('all_words:',len(all_words),'current_entropy:',current_entropy,all_words)
    # print('words_sorted_by_freq:',len(words_sorted_by_freq),words_sorted_by_freq)

    # Adjust the banned list until the entropy meets the threshold
    for word in words_sorted_by_freq:
        if current_entropy < threshold_entropy and word not in banned_list:
            banned_list.append(word)
            word_counts[word] = 1  # Assume word is not used when it's in banned list
        # elif current_entropy >= threshold_entropy and word in banned_list:
        #     banned_list.remove(word)
        #     word_counts[word] = 1  # Reset count for simplicity

        current_entropy = calculate_entropy(word_counts)
        # print('current_entropy:',current_entropy)
        if current_entropy >= threshold_entropy:
            break
    # print('banned_list:',len(banned_list),'current_entropy',current_entropy,banned_list)
    return banned_list

def cs_sampling(all_center_repre,k = 10):
    selected_center_idx = torch.randint(0, all_center_repre.size(0), (1,)).item()
    # print('selected_center_idx:',selected_center_idx)

    # Selected center representation
    selected_center = all_center_repre[selected_center_idx]

    # Calculate cosine similarity between the selected center and all centers
    cosine_similarities = F.cosine_similarity(selected_center.unsqueeze(0), all_center_repre)

    # Get indices of the 9 most similar centers (excluding the selected center itself)
    _, indices = torch.topk(cosine_similarities, k)
    # print('indices:',indices)
    indices = indices.detach().cpu()

    # Ensure the selected center index is included and sort the indices
    final_indices = torch.cat((torch.tensor([selected_center_idx]), indices))
    final_indices = torch.unique(final_indices, sorted=True)

    # Output the 10 center indices
    final_indices.tolist()
    return final_indices
