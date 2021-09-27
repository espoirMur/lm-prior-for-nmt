import torch
from torch import nn
from gensim.models import FastText
import numpy as np
from tqdm import tqdm
from pathlib import Path


class EmbeddingsReader:
    """
    this class is responsible for reading the embddiing file from different files format.s

    Returns:
        [type]: [description]
    """

    @staticmethod
    def from_text(filename, vocab, unif=0.25):
        
        with io.open(filename, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.rstrip("\n ")
                values = line.split(" ")

                if i == 0:
                    # fastText style
                    if len(values) == 2:
                        weight = init_embeddings(len(vocab), values[1], unif)
                        continue
                    # glove style
                    else:
                        weight = init_embeddings(len(vocab), len(values[1:]), unif)
                word = values[0]
                if word in vocab:
                    vec = np.asarray(values[1:], dtype=np.float32)
                    weight[vocab.tok2id.get(word)] = vec
        if '[PAD]' in vocab:
            weight[vocab['[PAD]']] = 0.0
        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        return embeddings, weight.shape[1]
    
    @staticmethod
    def from_binary_fasttext(filename, vocab, unif=0.25):

        """Read the binary model from fasttext
        vocab : instance of vocabulary

        Returns:
            [type]: [description]
        """

        vocab_size = len(vocab)
        filename = Path.cwd().joinpath(filename).__str__()
        model = FastText.load(filename)
        weight = np.zeros((vocab_size, model.vector_size))
        for word in ['<pad>', '<sos>', '<eos>']:
            weight[vocab.tok2id.get(word)] = 0.0
        for token_id, token in tqdm(vocab.id2tok.items(),
                                    desc="Reading embeddings...",
                                    total=len(vocab.id2tok.items())):
            vec = model.wv[token]
            weight[token_id] = vec
        embeddings = nn.Embedding(weight.shape[0], weight.shape[1])
        embeddings.weight = nn.Parameter(torch.from_numpy(weight).float())
        embeddings.weight.requires_grad = False
        return embeddings
