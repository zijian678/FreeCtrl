for sentence_thre in $(seq 1.10 0.02 1.2); do
  for sentence_thre_senti in $(seq 1.10 0.02 1.2); do
    for k_values in $(seq 150 50 250); do
      for lamda_topic in $(seq 0.3 0.1 1.0); do
        for lamda_senti in $(seq 0.3 0.1 1.0); do
          python -u multi_control.py \
            --sentence_thre $sentence_thre \
            --sentence_thre_senti $sentence_thre_senti \
            --k_values $k_values \
            --lamda_topic $lamda_topic \
            --lamda_senti $lamda_senti
        done
      done
    done
  done
done