import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
import argparse, csv
from utilis import *

# Basic setting
parser = argparse.ArgumentParser(description='FreeCtrl: sentiment control')
parser.add_argument('--model', type=str, default="gpt2-medium")
parser.add_argument('--device', type=str, default=torch.device('cuda:0'), help='Device where Model Trained on: gpu(default) | cpu')
parser.add_argument('--noise_threshold', type=float, default=1.0, help='threshold for de-noising')

# method hyperpatameters
parser.add_argument('--sentence_thre', type=float, default=1.15, help='sentence threshold u_w')
parser.add_argument('--k_values', type=int, default=30, help='maximum number of value vectors for each keyword')
parser.add_argument('--lamda', type=float, default=0.20, help='scaling parameter')
parser.add_argument('--label_word_file', type=str, default="senti_words_v2.txt", help='label keyword file')

# task setting
parser.add_argument('--label_id', type=int, default=1, help='label id for generation') # 0 - positive; 1 - negative.
parser.add_argument('--out_path', type=str, default='./output/')
parser.add_argument('--out_version', type=str, default='v1')

# generation papameters
parser.add_argument('--min_length', type=int, default=50)
parser.add_argument('--max_length', type=int, default=50)
parser.add_argument('--do_sample', type=bool, default=True)
parser.add_argument('--num_beams', type=int, default=5)
parser.add_argument('--top_k', type=int, default=25)
parser.add_argument('--num_return_sequences', type=int, default=5)
parser.add_argument('--temperature', type=float, default=100.0)

args = parser.parse_args()

print('Parameters:', 'sentence_thre:',args.sentence_thre,'k_values:',args.k_values,'lamda:',args.lamda)
## initialize parameters
device = args.device
tokenizer, model = load_model(device=device,model_name = args.model)


## initialize generation papameters
min_length = args.min_length
max_length = args.max_length
do_sample = args.do_sample
num_beams = args.num_beams
top_k = args.top_k
num_return_sequences = args.num_return_sequences
temperature = args.temperature


# tokenizer, model = load_model(device=device,model_name = args.model)
emb_weight = model.transformer.wte.weight # weight matrix: torch.Size([50257, 1024])
print('weight matrix:',emb_weight.shape)


## obtain class repre
class_name = ['positive','negative']
class_ids = tokenizer.convert_tokens_to_ids(class_name)
print('class_ids:',class_ids,tokenizer.convert_ids_to_tokens(class_ids))
class_repre = emb_weight[class_ids]
print('class_repre:',class_repre.shape,class_repre)

## evaluate diversity of each clustering map
wrapper = GPT2Wrapper(model_name = "gpt2-medium", use_cuda = True)


# save value vector positions
v_positions = []
start = 0
for v in range(24):
    for iiid in range(4096):
        v_positions.append((v,iiid))
num_samples = 98304 # 98304
prompt = ' '

control_probs = torch.load('../control_probs.pt')
control_probs = control_probs.float()

print('Filtering keywords for each class')
label_file = open(args.label_word_file,'r')
lines = label_file.readlines()
label_ids = {}
label_count = 0
all_label_repre = []
for line in lines:
    words = line.split(',')
    ids = []
    for w in words:
        cw = w.strip()
        cw = ' ' + cw
        cw_id = tokenizer.encode(cw)
        filter_socre = de_noising(class_repre,emb_weight[cw_id[0]],label_count,4)
        # print('filter_socre:',filter_socre,'token:',tokenizer.decode(cw_id[0]))
        if len(cw_id) < 2 and cw_id[0] != 50267 and filter_socre > args.noise_threshold:
            # print('cw_id:', cw_id, tokenizer.decode(cw_id))
            ids.append(cw_id[0])
    label_ids[label_count] = ids
    label_repre = emb_weight[ids]
    all_label_repre.append(label_repre)
    label_count = label_count + 1

key_ids = label_ids[args.label_id]
print('# of Filtered keywords:',len(key_ids),tokenizer.convert_ids_to_tokens(key_ids))
# load keywords_idx
# for lr in all_label_repre:
#     print(lr.shape)
keyword_tokens = []
for i in key_ids:
    keyword_tokens.append(tokenizer.decode(i).strip())
print('keyword_tokens:',keyword_tokens)

all_center_repre = all_label_repre[args.label_id]
# print('all_center_repre:',all_center_repre)

all_probs = control_probs[:,key_ids]
all_probs = all_probs.squeeze() # torch.Size([98304, 143])
all_probs = all_probs.t()


# act_num = args.k_values # number of activated neurons
# observe top tokens
all_control_centers = []
all_original_positions = []
# construct control vectors for each keyword
for prob, t_id in zip(all_probs,key_ids):
    # print('*************************** current keyword:',tokenizer.convert_ids_to_tokens([t_id]),'*****************************')
    kv_values, kv_pos = torch.topk(prob, k= args.k_values)
    # print('kv_values:', kv_values)

    current_values = [v_positions[i] for i in kv_pos]
    all_original_positions.append(current_values)
    #     current_values = organize_list(current_values)
    current_values = organize_list(current_values)
    all_control_centers.append(current_values)

prompts = ['In summary', 'This essay discusses', 'Views on', 'The connection', 'Foundational to this is',
 'To review,', 'In brief,', 'An illustration of', 'Furthermore,', 'The central theme',
 'To conclude,', 'The key aspect', 'Prior to this', 'Emphasised are', 'To summarise',
 'The relationship', 'More importantly,', 'It has been shown', 'The issue focused on', 'In this essay',
 'Once upon a time', 'The book', 'The chicken', 'The city', 'The country',
 'The horse', 'The lake', 'The last time', 'The movie', 'The painting',
 'The pizza', 'The potato', 'The president of the country', 'The road', 'The year is 1910']


# prompts = ['In summary']

total_l = max_length
step = 1 # token generating step
all_gen = []
all_used_tokens = keyword_tokens
banned_list = ['none']
all_choices = len(keyword_tokens)



for prompt in prompts:

    each_prompt_gen = []
    trial_number = 0

    current_pos = []
    for idx in range(all_choices):
        pos = all_original_positions[idx]
        # target_repre.append(all_center_repre[idx].unsqueeze(0))
        for xxy in pos:
            current_pos.append(xxy)
    current_control = organize_list(current_pos)


    while len(each_prompt_gen) < (args.num_return_sequences):
        trial_number = trial_number + 1
        current_input = prompt
        length = len(tokenizer.encode(current_input))
        ori_length = length
        # print('original prompt:',current_input,'original_length:',length)
        max_length = length + total_l
        all_weight = []

        old_weight = 0.0

        while length < max_length:
            ot_ids = tokenizer.encode(current_input)
            ot_repre = emb_weight[ot_ids]
            sen_score_topic = scoring(all_label_repre, ot_repre, label_id=args.label_id, label_num=len(class_name))
            token_score_topic = scoring(all_label_repre, ot_repre[-1, :].unsqueeze(0),
                                        label_id=args.label_id, label_num=len(class_name))
            final_score = torch.max(sen_score_topic, token_score_topic)

            if args.sentence_thre - final_score > 0:
                new_weight = args.lamda / (1.0 + torch.exp(-torch.tensor(args.sentence_thre - final_score) * (length)))
            else:
                new_weight = 1e-20

            # print('new_weight 0:',new_weight)
            new_weight = (new_weight) #+ old_weight)/2.0
            # print('old_weight:',old_weight,'new_weight 1:', new_weight)
            old_weight = new_weight

            # if args.sentence_thre - final_score < 1e-20:
            #     new_weight = 1e-20

            all_weight.append(float(new_weight))

            wrapper.set_value_activations(current_control, coef_value=new_weight)

            try:
                output_texts = wrapper.generate([current_input],

                                                            # min_length=step,
                                                            max_length=step,
                                                            do_sample=do_sample,
                                                            num_beams=num_beams,
                                                            top_k=top_k,
                                                            temperature=temperature,
                                                            num_return_sequences=1)
            except:
                output_texts = [" "]

            wrapper.remove_all_hooks()
            current_input = current_input + output_texts[0]
            length = len(tokenizer.encode(current_input))

        split_ = current_input.split('\n')
        if len(split_) > 2:
            continue
        ot_ids = tokenizer.encode(current_input)
        ot_repre = emb_weight[ot_ids]

        # final_score = de_noising(final_cls_repre, s_mean, label_id=args.label_id, label_num=4)
        final_score = scoring(all_label_repre, ot_repre, label_id=args.label_id, label_num=len(class_name))
        sentence_s = final_score
        # print('original sentence_s:',sentence_s)

        max_count = Counter(current_input.split())
        word_with_max_freq = max(max_count, key=max_count.get)

        # Returning the word and its frequency
        max_freq =  max_count[word_with_max_freq]
        # print('max_freq:',max_freq)
        print('sentence_s:', sentence_s)


        if sentence_s > (args.sentence_thre ) and max_freq < 7:  # max_freq remove trash outputs, not necessary
            print('***********************************************')
            print('sentence_s:',sentence_s)
            print(current_input)

            all_gen.append(current_input)
            each_prompt_gen.append(current_input)

            banned_list = ['none']

            # print('all_used_tokens:',len(all_used_tokens),all_used_tokens)
            print('number:',len(all_gen),'sentence_s',sentence_s)

        if trial_number > 50:  # avoid dead loop
            print('exceeding max trial, exit')
            break


    #
#
out_path = args.out_path + args.out_version + '_' + 'la'+str(args.lamda)+ 't' + str(args.sentence_thre)+'_' + str(args.label_id) + '.csv'

#sentence_thre', type=float, default=1.10, help='threshold for de-noising')
# parser.add_argument('--relaxation

with open(out_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Writing each sentence as a new row
    for sentence in all_gen:
        writer.writerow([sentence])

print('Results saved to ',out_path)

