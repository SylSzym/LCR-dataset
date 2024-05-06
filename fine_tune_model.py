import pandas as pd
import numpy as np
import torch
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
from transformers import TrainingArguments, Trainer
from transformers import BertTokenizer, BertForSequenceClassification

class DataProcessor:
    def __init__(self, file_path, sep=';'):
        self.file_path = file_path
        self.sep = sep

    def load_data(self):
        data = pd.read_csv(self.file_path, sep=self.sep)
        data = data.dropna()
        data = data.loc[data['Category'].isin(['LCR_with_function', 'hard_0'])]
        data['text'] = data['Title'] + ' ' + data['Abstract']
        data['labels'] = 0
        data.loc[data.Category == 'LCR_with_function', 'labels'] = 1
        return data[['text', 'labels']]

class Tokenizer:
    def __init__(self, model_name):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

    def tokenize(self, texts, padding=True, truncation=True, max_length=512):
        return self.tokenizer(texts, padding=padding, truncation=truncation, max_length=max_length)

class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels = None):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        if self.labels:
            item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.encodings["input_ids"])

class ModelTrainer:
    def __init__(self, model, args, train_dataset, eval_dataset):
        self.trainer = Trainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            compute_metrics=self.compute_metrics
        )

    def compute_metrics(self, p):
        pred, labels = p
        pred = np.argmax(pred, axis=1)

        accuracy = accuracy_score(y_true=labels, y_pred=pred)
        recall = recall_score(y_true=labels, y_pred=pred)
        precision = precision_score(y_true=labels, y_pred=pred)
        f1 = f1_score(y_true=labels, y_pred=pred)

        return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1": f1}

    def train(self):
        self.trainer.train()

    def evaluate(self):
        self.trainer.evaluate()

    def save_model(self, output_dir):
        self.trainer.save_model(output_dir)

def main():
    data_processor = DataProcessor('./sample_data/raw_data/training_file.csv')
    data = data_processor.load_data()

    data_val_processor = DataProcessor('./sample_data/raw_data/validation_file.csv')
    data_val = data_val_processor.load_data()

    tokenizer = Tokenizer("google-bert/bert-base-uncased")
    X_train, y_train = list(data["text"]), list(data["labels"])
    X_val, y_val = list(data_val["text"]), list(data_val["labels"])

    X_train_tokenized = tokenizer.tokenize(X_train)
    X_val_tokenized = tokenizer.tokenize(X_val)

    train_dataset = Dataset(X_train_tokenized, y_train)
    val_dataset = Dataset(X_val_tokenized, y_val)

    model = BertForSequenceClassification.from_pretrained("google-bert/bert-base-uncased", num_labels=2)

    args = TrainingArguments(
        output_dir="output/model",
        num_train_epochs=4,
        per_device_train_batch_size=8
    )

    model_trainer = ModelTrainer(model, args, train_dataset, val_dataset)
    model_trainer.train()
    model_trainer.evaluate()
    model_trainer.save_model('./output/model/B_PRE_BERT_P_h0')

if __name__ == "__main__":
    main()