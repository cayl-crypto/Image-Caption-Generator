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
        
