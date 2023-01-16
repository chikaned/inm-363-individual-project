
def GetInvDict(embedding_file_path = './embedding/pretrained_embeddings/glove_wiki_100d.txt'):
    """

    Get the inverted dictionary that can be used to generate words from ints using embedding word2idx (GloVe wiki 100d used by default)

    """
    #read_path
    fin = open(embedding_file_path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    stoi = {}    

    #load pretrained vectors
    for i, line in enumerate(fin):
        tokens = line.rstrip().split(' ')
        word = tokens[0]
        stoi[i]=word

    return stoi