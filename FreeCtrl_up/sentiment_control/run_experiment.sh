#python -u single_topic_control.py --label_id 0 --sentence_thre 1.07 --lamda 0.30


for label_id in {0..1}; do
  for sentence_thre in $(seq 1.04 0.02 1.2); do
    for lamda in $(seq 0.1 0.2 1.1); do
      python -u single_topic_control.py --label_id $label_id --sentence_thre $sentence_thre --lamda $lamda
    done
  done
done
