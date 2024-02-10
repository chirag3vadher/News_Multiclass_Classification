import re
import pandas as pd
from config import training_data
from Utils.Finetune.finetuning import Triage
from transformers import DistilBertModel, DistilBertTokenizer
from nltk.corpus import stopwords
from torch.utils.data import DataLoader


class loader():
    def load_model_and_tokenizer(model_path, tokenizer_path):
        # Load the model
        model = DistilBertModel.from_pretrained(model_path)
        # Load the tokenizer
        tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_path)
        return model, tokenizer

    def read_training_file(file_path):
        df = pd.read_csv(file_path)
        return df

class pre_processor():
    def remove_punctuations(text):
        text = re.sub(r'[\\-]', ' ', text)
        text = re.sub(r'[,.?;:\'(){}!|0-9]', '', text)
        return text
    def remove_stopwords(text):
        clean_text = []
        stopw = stopwords.words('english')
        for word in text.split(' '):
            if word not in stopw:
                clean_text.append(word)
        return ' '.join(clean_text)

    def preprocess_data(tokenizer, MAX_LEN, TRAIN_BATCH_SIZE, VALID_BATCH_SIZE):

        # Read the dataframe
        df = pd.read_csv(training_data)

        # Rename 'Class Index' column to 'label'
        df = df.rename(columns={'Class Index': 'label'})

        # Concatenate 'Title' and 'Description' columns to create 'text'
        df['text'] = df['Title'] + df['Description']

        # Mapping numeric labels to categories
        category_map = {1: "World", 2: "Sports", 3: "Business", 4: "Sci/Tech"}
        df['category'] = df["label"].map(category_map)

        # Encoding categories
        encode_dict = {}
        def encode_cat(x):
            if x not in encode_dict:
                encode_dict[x] = len(encode_dict)
            return encode_dict[x]

        df['ENCODE_CAT'] = df['category'].apply(encode_cat)

        # Splitting into train and test datasets
        train_size = 0.8
        train_dataset = df.sample(frac=train_size, random_state=200)
        test_dataset = df.drop(train_dataset.index).reset_index(drop=True)
        train_dataset.reset_index(drop=True, inplace=True)

        # Printing dataset sizes
        print("FULL Dataset: {}".format(df.shape))
        print("TRAIN Dataset: {}".format(train_dataset.shape))
        print("TEST Dataset: {}".format(test_dataset.shape))

        # Assuming Triage is a class for dataset creation
        training_set = Triage(train_dataset, tokenizer, MAX_LEN)
        testing_set = Triage(test_dataset, tokenizer, MAX_LEN)

        # Data loader parameters
        train_params = {'batch_size': TRAIN_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}
        test_params = {'batch_size': VALID_BATCH_SIZE, 'shuffle': True, 'num_workers': 0}

        # Creating data loaders
        training_loader = DataLoader(training_set, **train_params)
        testing_loader = DataLoader(testing_set, **test_params)

        print(type(training_loader))

        return training_loader, testing_loader

