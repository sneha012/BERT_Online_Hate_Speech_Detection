import numpy as np

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset, SequentialSampler

from sklearn.preprocessing import OneHotEncoder
from transformers import DistilBertModel, DistilBertTokenizer, AutoTokenizer

from tqdm import tqdm


DISTIL_MODEL_NAME = "distilbert-base-uncased"  # "distilbert-base-uncased-finetuned-sst-2-english"
DISTIL_TOKENIZER = DistilBertTokenizer.from_pretrained(DISTIL_MODEL_NAME, truncation=True, do_lower_case=True)
DISTIL_TRAIN_BATCH_SIZE = 16
DISTIL_DEV_BATCH_SIZE = 1
DISTIL_TEST_BATCH_SIZE = 1

def get_distil_hyperparams():
    return {
        "MAX_LEN": 128,
        "LEARNING_RATE": 1e-05,
        "TOKENIZER": DISTIL_TOKENIZER,
        "DEVICE": torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
        "TRAIN_PARAMS": {
            "batch_size": DISTIL_TRAIN_BATCH_SIZE,
            "shuffle": True,
            "num_workers": 0
        },
        "DEV_PARAMS": {
            "batch_size": DISTIL_DEV_BATCH_SIZE,
            "shuffle": False,
            "num_workers": 0
        },
        "TEST_PARAMS": {
            "batch_size": DISTIL_TEST_BATCH_SIZE,
            "shuffle": False,
            "num_workers": 0
        },
        "MODEL_PATH": "../models/pytorch_distilbert.bin",
        "VOCAB_PATH": "../models/vocab_distilbert.bin"
    }


def get_bertweet_hyperparams():
    return {
        "MODEL_NAME": "vinai/bertweet-large",
        "BATCH_SIZE": 1,
        "MODEL_PATH": "../models/bertweet_large_pkl"
    }


def get_roberta_hyperparams():
    return {
        "MODEL_PATH": "../models/roberta.model"
    }

class HateDataset(Dataset):

    def __init__(self, dataframe, tokenizer, max_len):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.text = dataframe.text
        self.targets = OneHotEncoder(sparse=False).fit_transform(np.array(self.data["hatespeech"]).reshape(-1, 1))
        self.max_len = max_len

    def __len__(self):
        return len(self.text)

    def __getitem__(self, index):
        text = str(self.text[index])
        text = " ".join(text.split())

        inputs = self.tokenizer.encode_plus(
            text,
            None,
            add_special_tokens=True,
            max_length=self.max_len,
            pad_to_max_length=True,
            return_token_type_ids=True
        )

        ids = inputs["input_ids"]
        mask = inputs["attention_mask"]
        token_type_ids = inputs["token_type_ids"]

        return {
            "ids": torch.tensor(ids, dtype=torch.long),
            "mask": torch.tensor(mask, dtype=torch.long),
            "token_type_ids": torch.tensor(token_type_ids, dtype=torch.long),
            "targets": torch.tensor(self.targets[index], dtype=torch.float)
        }

class DistilBERTMultiClass(nn.Module):
    def __init__(self, n_classes, model_name=DISTIL_MODEL_NAME):
        super(DistilBERTMultiClass, self).__init__()
        self.l1 = DistilBertModel.from_pretrained(model_name)
        self.pre_classifier = nn.Linear(768, 768)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(768, n_classes)

    def forward(self, input_ids, attention_mask, token_type_ids):
        output_1 = self.l1(input_ids=input_ids, attention_mask=attention_mask)
        hidden_state = output_1[0]
        pooler = hidden_state[:, 0]
        pooler = self.pre_classifier(pooler)
        pooler = nn.Tanh()(pooler)
        pooler = self.dropout(pooler)
        output = self.classifier(pooler)
        return output


def predict_distilbert(model, loader, device):
    model.to(device)
    model.eval()
    fin_outputs = []
    with torch.no_grad():
        for _, data in tqdm(enumerate(loader)):
            ids = data["ids"].to(device, dtype=torch.long)
            mask = data["mask"].to(device, dtype=torch.long)
            token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
            outputs = model(ids, mask, token_type_ids)
            fin_outputs.extend(torch.sigmoid(outputs).cpu().detach().numpy().tolist())
    return fin_outputs


def encode_data(df, tokenizer):
    input_ids = []
    attention_masks = []
    for tweet in df[["text"]].values:
        tweet = tweet.item()
        encoded_data = tokenizer.encode_plus(
                            tweet,                      
                            add_special_tokens = True,  
                            max_length = 128,
                            padding = 'max_length',
                            truncation = True,
                            return_attention_mask = True,   
                            return_tensors = 'pt',    
                    )
        input_ids.append(encoded_data['input_ids'])
        attention_masks.append(encoded_data['attention_mask'])
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)

    inputs = {
        'input_ids': input_ids,
        'input_mask': attention_masks
    }
    return inputs


def prepare_dataloaders(test_df, model_name, batch_size):
    tokenizer = AutoTokenizer.from_pretrained(model_name,use_fast=False, normalization=True)

    data_test = encode_data(test_df, tokenizer)

    input_ids, attention_masks = data_test.values()
    test_dataset = TensorDataset(input_ids, attention_masks)

    test_dataloader = DataLoader(
        test_dataset, 
        sampler = SequentialSampler(test_dataset), 
        batch_size = batch_size
    )

    return test_dataloader


def predict_bert_tweet_roberta(model, loader, device):
    model.to(device)
    model.eval()
    preds = []

    for batch in loader:
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        with torch.no_grad():        
            outputs = model(
                b_input_ids, 
                token_type_ids=None, 
                attention_mask=b_input_mask
            )
            logits = outputs.logits

        logits = logits.detach().cpu().numpy()
        preds.extend(logits)

    return preds