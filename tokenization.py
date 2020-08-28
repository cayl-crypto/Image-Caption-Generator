from tqdm import tqdm

def tokenize(vocabulary, captions):
    
    print()
    print("Tokenizing captions...")
    tokenized_captions = []
    for caption in tqdm(captions):
        caption_tokens = []
        for word in caption.split(" "):
            if word in vocabulary.word2index:
                caption_tokens.append(vocabulary.word2index[word])
           
        tokenized_captions.append(caption_tokens)
    return tokenized_captions
        

def get_maximum_length_of_captions(tokenized_captions):
    max_len = 0
    for tokenized_caption in tqdm(tokenized_captions):
        len_cap = len(tokenized_caption)
        if len_cap > max_len:
            max_len = len_cap
    return max_len


def pad_sequences(sequences):
    pass