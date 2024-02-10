from transformers import DistilBertTokenizer, DistilBertModel
from torch import cuda
import os

current_directory = os.getcwd()

# Setting up the device for GPU usage
device = 'cuda' if cuda.is_available() else 'cpu'

#defining the tokenizer and model
tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-cased')
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

# Loading the pretrained model and tokenizer from the path
model_path = "Model\pytorch_distilbert_news.bin"
tokenizer_path = "vocab_distilbert_news.bin"

#defining the training data path
training_data = f"{current_directory}\Data\\train.csv"