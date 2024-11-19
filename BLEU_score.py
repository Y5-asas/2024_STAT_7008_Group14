import pandas as pd
from nltk.translate.bleu_score import corpus_bleu, SmoothingFunction,sentence_bleu


def calculate_bleu_score(cal_data_path, source_language, target_language, translate, model, source_word_2_index, target_word_2_index, start_symbol):
    data = pd.read_csv(cal_data_path, usecols=lambda col: col != 'Unnamed: 0')
    
    # Split the sentences into words
    reference = [[sentence.split()] for sentence in data[target_language]]
    _src = [sentence for sentence in data[source_language]]
    
    # Translate the source sentences and remove <SOS>, <EOS>, and <PAD> tokens
    candidate = []
    i=0
    for sentence in _src:
        translated_sentence = translate(
            model=model,
            src=sentence,
            source_word_2_index=source_word_2_index,
            target_word_2_index=target_word_2_index,
            start_symbol=start_symbol
        ).split()
        # Filter out <SOS>, <EOS>, and <PAD> tokens
        filtered_sentence = []
        sos_found = False
        for word in translated_sentence:
            if word == '<SOS>':
                sos_found = True
                continue
            if word == '<EOS>':
                break
            if word != '<PAD>':
                if sos_found:
                    filtered_sentence.append(word)
                elif not sos_found and word != '<SOS>':
                    filtered_sentence.append(word)
        candidate.append(filtered_sentence)
    # Calculate BLEU score
    smoothing = SmoothingFunction().method4
    # print(f"reference:\n {reference}")
    # print(f"candidate:\n {candidate}")
    bleu_score = corpus_bleu(reference, candidate, smoothing_function=smoothing)
    return bleu_score

'''
# example usage
file_path = './nusax-main/datasets/mt/valid.csv'
source_language = 'indonesian'
target_language = 'english'
model = ... 
source_word_2_index = built_curpus(data[args.source_language])
target_word_2_index = built_curpus(data[args.target_language])
def translate(model, src, source_word_2_index, target_word_2_index, start_symbol=2):
    src = src_sentence              # a sentence in source language
    translated_sentence = ...       # a sentence in target language(with <SOS>, <EOS> and <PAD> tokens)
    return translated_sentence

score = calculate_bleu_score(test_file, src_language, tgt_language, translate, model, source_word_2_index, target_word_2_index,2)
'''