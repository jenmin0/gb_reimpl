from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
from nltk.util import ngrams
from nltk.corpus import brown
import dill as pickle
import pandas as pd


from  ghostbuster.utils import featurize,n_gram,symbolic



trigram_model = symbolic.train_trigram()
pickle.dump(trigram_model, open("ghostbuster/model/trigram_model.pkl", "wb"))  
