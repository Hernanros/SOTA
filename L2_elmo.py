from flair.embeddings import ELMoEmbeddings
from flair.data import Sentence
from scipy.spatial.distance import euclidean
import torch

# init embedding
embedding = ELMoEmbeddings()

# create a sentence
sentence1 = Sentence('The grass is green .')
sentence2 = Sentence('Across the yard, there is green grass .')

# embed words in sentence
embedding.embed(sentence1)
embedding.embed(sentence2)

embed1 = torch.cat([token.embedding for token in sentence1])
embed2 = torch.cat([token.embedding for token in sentence2])
print(euclidean(embed1, embed2))

