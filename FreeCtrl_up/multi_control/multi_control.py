import os
os.environ['CUDA_VISIBLE_DEVICES'] = "0"
os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
import argparse, csv
from utilis import *




parser = argparse.ArgumentParser(description='FreeCtrl: Multi-Attribute Control')
parser.add_argument('--model', type=str, default="gpt2-medium")
parser.add_argument('--device', type=str, default=torch.device('cuda'), help='Device where Model Trained on: gpu(default) | cpu')

# method hyperpatameters - topic
parser.add_argument('--noise_threshold', type=float, default=1.0, help='threshold for de-noising')
# parser.add_argument('--sentence_thre_topic', type=float, default=1.13, help='threshold for de-noising')
parser.add_argument('--sentence_thre', type=float, default=1.10, help=' ')
parser.add_argument('--sentence_thre_senti', type=float, default=1.10, help=' ')
parser.add_argument('--k_values', type=int, default=200, help='maximum number of value vectors for each keyword')
parser.add_argument('--lamda_topic', type=float, default=0.3, help='scaling parameter')
parser.add_argument('--lamda_senti', type=float, default=0.3, help='scaling parameter')

# task setting
parser.add_argument('--topic_label_id', type=int, default=1, help='label id for topic generation from 0 to 3')
parser.add_argument('--sentiment_label_id', type=int, default=0, help='label id for sentiment generation from 0 to 1')
parser.add_argument('--out_path', type=str, default='./output/')
parser.add_argument('--out_version', type=str, default='v1')

# generation papameters
parser.add_argument('--min_length', type=int, default=50)
parser.add_argument('--max_length', type=int, default=50)
parser.add_argument('--do_sample', type=bool, default=True)
parser.add_argument('--num_beams', type=int, default=20)
parser.add_argument('--top_k', type=int, default=30)
parser.add_argument('--num_return_sequences', type=int, default=5)
parser.add_argument('--temperature', type=float, default=200.0)

args = parser.parse_args()


## initialize parameters
device = args.device
tokenizer, model = load_model(device=device,model_name = args.model)


## initialize generation papameters
# coef_value = args.coef
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

all_class_name = [['politics','sports','business','technology'],['positive','negative']]
all_keyword_path = ['topic_words.txt','senti_words_v2.txt']
all_label_ids = [args.topic_label_id,args.sentiment_label_id]
task_all_original_positions = []
task_keyword_tokens = []
task_key_ids = []
task_all_label_repre = []
for class_name,keyword_path,label_id in zip(all_class_name,all_keyword_path,all_label_ids):
    ## obtain class repre
    class_ids = tokenizer.convert_tokens_to_ids(class_name)
    print('class_ids:',class_ids,tokenizer.convert_ids_to_tokens(class_ids))
    class_repre = emb_weight[class_ids]
    print('class_repre:',class_repre.shape,class_repre)

    # save v_positions
    v_positions = []
    start = 0
    for v in range(24):
        for iiid in range(4096):
            v_positions.append((v,iiid))
        #     iiid = iiid + 1
        # start = start + 1

    num_samples = 98304 # 98304
    prompt = ' '

    control_probs = torch.load('../control_probs.pt')
    control_probs = control_probs.float()

    print('Filtering keywords for each class')
    label_file = open(keyword_path,'r')
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
    task_all_label_repre.append(all_label_repre)

    key_ids = label_ids[label_id]
    task_key_ids.append(key_ids)
    print('# of Filtered keywords:',len(key_ids),tokenizer.convert_ids_to_tokens(key_ids)[:10])
    # load keywords_idx
    # for lr in all_label_repre:
    #     print(lr.shape)
    keyword_tokens = []
    for i in key_ids:
        keyword_tokens.append(tokenizer.decode(i).strip())
    print('keyword_tokens:',len(keyword_tokens),keyword_tokens[:10])
    task_keyword_tokens.append(keyword_tokens)


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
    task_all_original_positions.append(all_original_positions)

print('task_all_original_positions:',len(task_all_original_positions),len(task_all_original_positions[0]),len(task_all_original_positions[1]))
print('task_keyword_tokens',len(task_keyword_tokens),len(task_keyword_tokens[0]),len(task_keyword_tokens[1]))
print('task_key_ids:',len(task_key_ids),len(task_key_ids[0]),len(task_key_ids[1]))

prompts = ['In summary', 'This essay discusses', 'Views on', 'The connection', 'Foundational to this is',
 'To review,', 'In brief,', 'An illustration of', 'Furthermore,', 'The central theme',
 'To conclude,', 'The key aspect', 'Prior to this', 'Emphasised are', 'To summarise',
 'The relationship', 'More importantly,', 'It has been shown', 'The issue focused on', 'In this essay',
 'Once upon a time', 'The book', 'The chicken', 'The city', 'The country',
 'The horse', 'The lake', 'The last time', 'The movie', 'The painting',
 'The pizza', 'The potato', 'The president of the country', 'The road', 'The year is 1910']
#
#
# prompts = ['In summary']

# 0.5 * torch.exp(torch.tensor(15 * (0.5 - 0.2) / 2))

# for attribute_num in range(len(task_key_ids)):
total_l = max_length
step = 1 # token generating step
all_gen = []
all_used_tokens = keyword_tokens
banned_list = ['none']
all_choices_topic = len(keyword_tokens[0])
all_choices_senti = len(keyword_tokens[1])
wrapper = GPT2Wrapper(model_name = "gpt2-medium", use_cuda = True)

# used parameters: all_choices, all_original_positions,all_label_repre,label_id，label_num

# build up centers for each attribute


for prompt in prompts:

    each_prompt_gen = []
    trial_number = 0

    current_pos_topic = []
    for idx in range(all_choices_topic):
        pos = task_all_original_positions[0][idx]
        # target_repre.append(all_center_repre[idx].unsqueeze(0))
        for xxy in pos:
            current_pos_topic.append(xxy)
    current_control_topic = organize_list(current_pos_topic)
    all_label_repre_topic = task_all_label_repre[0]

    current_pos_senti = []
    for idx in range(all_choices_senti):
        pos = task_all_original_positions[1][idx]
        # target_repre.append(all_center_repre[idx].unsqueeze(0))
        for xxy in pos:
            current_pos_senti.append(xxy)
    current_control_senti = organize_list(current_pos_senti)
    all_label_repre_senti = task_all_label_repre[1]


    while len(each_prompt_gen) < (args.num_return_sequences):
        trial_number = trial_number + 1
        current_input = prompt
        length = len(tokenizer.encode(current_input))
        ori_length = length
        # print('original prompt:',current_input,'original_length:',length)
        max_length = length + total_l
        all_weight_topic = []
        all_weight_senti = []
        old_weight_topic = 0.0
        old_weight_senti = 0.0


        while length < max_length:
            ot_ids = tokenizer.encode(current_input)
            ot_repre = emb_weight[ot_ids]


            ## start organizing weights and attributes

            # control topic
            sen_score_topic = scoring(all_label_repre_topic, ot_repre, label_id=args.topic_label_id, label_num=4)
            token_score_topic = scoring(all_label_repre_topic, ot_repre[-1,:].unsqueeze(0), label_id=args.topic_label_id, label_num=4)
            final_score_topic = torch.max(sen_score_topic,token_score_topic)
            # print(sen_score_topic,token_score_topic,final_score_topic)

            # new_weight_topic = 1.0 / (1.0 + torch.exp(-torch.tensor(args.sentence_thre - final_score_topic) * (length ** 3)))
            new_weight_topic = args.lamda_topic / (1.0 + torch.exp(-torch.tensor(args.sentence_thre - final_score_topic) * (length)))


            if args.sentence_thre - final_score_topic <0.0:
                new_weight_topic = 1e-20

            new_weight_topic = (new_weight_topic) #+ old_weight_topic)/2.0
            old_weight_topic = new_weight_topic


            # control sentiment
            # print('ot_repre:',ot_repre.shape) # ot_repre: torch.Size([22, 1024])
            sen_score_senti = scoring(all_label_repre_senti, ot_repre, label_id=args.sentiment_label_id, label_num=2)
            token_score_senti = scoring(all_label_repre_senti, ot_repre[-1,:].unsqueeze(0), label_id=args.sentiment_label_id, label_num=2)
            final_score_senti = torch.max(sen_score_senti, token_score_senti)

            # new_weight_senti = 2.0 / (
            #             1.0 + torch.exp(-torch.tensor(args.sentence_thre - final_score_senti) * (length ** 3)))

            new_weight_senti = args.lamda_senti / (1.0 + torch.exp(-torch.tensor(args.sentence_thre_senti - final_score_senti) * (length)))

            if args.sentence_thre_senti - final_score_senti < 1e-20:
                new_weight_senti = 1e-20

            new_weight_senti = (new_weight_senti )#+ old_weight_senti)/2.0
            old_weight_senti = new_weight_senti

            if new_weight_senti > new_weight_topic:
                new_weight_topic = 1e-20
            else:
                new_weight_senti = 1e-20

            # if length > max_length/2:
            #     new_weight_topic = 1e-20
            # else:
            #     new_weight_senti = 1e-20





            wrapper.set_value_activations2(current_control_senti, new_weight_senti, current_control_topic, new_weight_topic)
            all_weight_topic.append(float(new_weight_topic))
            all_weight_senti.append(float(new_weight_senti))


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
            # print('new weight:',new_weight,'generated token:',output_texts[0])

            wrapper.remove_all_hooks()
            # wrapper.remove_all_hooks()
            current_input = current_input + output_texts[0]
            length = len(tokenizer.encode(current_input))

        sentence_topic_s = sen_score_topic
        sentence_senti_s = sen_score_senti
        # print('sentence score:',sentence_s)
        split_ = current_input.split('\n')
        if len(split_) > 2:
            continue

        # all_weight = torch.tensor(all_weight)
        # all_weight_mean = torch.mean(all_weight)
        print('sentence_topic_s:',sentence_topic_s,'sentence_senti_s:',sentence_senti_s)

        max_count = Counter(current_input.split())
        word_with_max_freq = max(max_count, key=max_count.get)

        # Returning the word and its frequency
        max_freq = max_count[word_with_max_freq]
        print('max_freq:', max_freq,current_input)


        if sentence_topic_s > (args.sentence_thre) and sentence_senti_s > (args.sentence_thre_senti) and max_freq < 7:  ## Can multiply these two as one hyper-parameter
            print('***********************************************')
            print('sentence_topic_s:',sentence_topic_s,'sentence_senti_s:',sentence_senti_s)
            # print('all_weight_topic:',torch.tensor(all_weight_topic))
            # print('all_weight_senti:', torch.tensor(all_weight_senti))
            # current_input = current_input.split('\n')[0]
            print(current_input)

            all_gen.append(current_input)
            each_prompt_gen.append(current_input)


            for i in list(set(current_input.split())):
                if i in keyword_tokens:
                    all_used_tokens.append(i.strip())
            # # all_used_tokens = list(set(all_used_tokens))
            # if banned_list == []:
            #     banned_list = ['none']
            # else:
            #     banned_list = update_ban(all_used_tokens, threshold_entropy = args.m_entropy)
            #     if banned_list == []:
            banned_list = ['none']

            # print('all_used_tokens:',len(all_used_tokens),all_used_tokens)
            print('number:',len(all_gen))

        if trial_number > 100:
            print('exceeding max trial, exit')
            break



    #
#
out_path = args.out_path + args.out_version + '_' + 'la'+str(args.lamda_topic)+'_'+str(args.lamda_senti) + 't' + str(args.sentence_thre) + '_' + str(args.topic_label_id) + '_' + str(args.sentiment_label_id) +'.csv'

#sentence_thre', type=float, default=1.10, help='threshold for de-noising')
# parser.add_argument('--relaxation

with open(out_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)

    # Writing each sentence as a new row
    for sentence in all_gen:
        writer.writerow([sentence])

print('Results saved to ',out_path)


