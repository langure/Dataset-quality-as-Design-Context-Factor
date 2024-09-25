# GLOBAL IMPORTS
import os

import pandas as pd
from tqdm import tqdm

# GLOBAL VARIABLES
SOURCE_DATA_PATH = './workdir'
TRAIN_DATA_PATH = os.path.join(SOURCE_DATA_PATH, 'train_dataframe.csv')
VALIDATION_DATA_PATH = os.path.join(SOURCE_DATA_PATH, 'validation_dataframe.csv')
RECALCULATE_DATASETS = True
WORK_DIR = './workdir/cli'

if not os.path.exists(WORK_DIR):
    os.makedirs(WORK_DIR)
    print(f"Directory {WORK_DIR} created successfully!")
else:
    print(f"Directory {WORK_DIR} already exists.")

file_paths = [TRAIN_DATA_PATH, VALIDATION_DATA_PATH]

for file_path in file_paths:
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        exit(1)

print("All files found!")


def calculate_cli(text):
    characters = len(text)
    words = len(text.split())
    sentences = len(text.split('.'))
    if sentences == 0:
        sentences = 1
    L = characters / words * 100
    S = sentences / words * 100
    cli = 0.058 * L - 0.296 * S - 15.8
    return cli


def assign_cli_to_examples(df, text_column='text'):
    df['cli'] = df[text_column].apply(calculate_cli)
    return df


def calculate_average_cli(df):
    average_cli = df['cli'].mean()
    return average_cli


def calculate_dataset_cli(file_path, text_column='text'):
    df = pd.read_csv(file_path)
    df.dropna(subset=['text'], inplace=True)
    df['cli'] = df[text_column].apply(calculate_cli)
    return df['cli'].mean()


def redistribute_cli(df, sample_size):
    min_count = df['shared_emotion'].value_counts().min() // 3
    sample_size = min(sample_size, min_count)

    high_cli_df = pd.DataFrame(columns=df.columns)
    mid_cli_df = pd.DataFrame(columns=df.columns)
    low_cli_df = pd.DataFrame(columns=df.columns)

    for emotion in df['shared_emotion'].unique():
        subset = df[df['shared_emotion'] == emotion].copy()
        subset = subset.sort_values('cli', ascending=False)  # Sort by CLI

        high_third = subset.iloc[:len(subset) // 3, :]
        mid_third = subset.iloc[len(subset) // 3:2 * len(subset) // 3, :]
        low_third = subset.iloc[2 * len(subset) // 3:, :]

        high_cli_df = high_cli_df.append(high_third.sample(sample_size))
        mid_cli_df = mid_cli_df.append(mid_third.sample(sample_size))
        low_cli_df = low_cli_df.append(low_third.sample(sample_size))

    return high_cli_df, mid_cli_df, low_cli_df


def create_and_save_datasets(directory, iterations, df, sample_size):
    for subfolder in ['high', 'mid', 'low']:
        os.makedirs(os.path.join(directory, subfolder), exist_ok=True)

    for i in tqdm(range(1, iterations + 1)):
        high_df, mid_df, low_df = redistribute_cli(df.copy(), sample_size)

        high_df.to_csv(os.path.join(directory, 'high', f'high_cli_{i}.csv'), index=False)
        mid_df.to_csv(os.path.join(directory, 'mid', f'mid_cli_{i}.csv'), index=False)
        low_df.to_csv(os.path.join(directory, 'low', f'low_cli_{i}.csv'), index=False)

    create_summary_csv(directory, 'high', iterations)
    create_summary_csv(directory, 'mid', iterations)
    create_summary_csv(directory, 'low', iterations)


def create_summary_csv(directory, cli_level, iterations):
    results = []
    for i in range(1, iterations + 1):
        filename = os.path.join(directory, cli_level, f'{cli_level}_cli_{i}.csv')
        df = pd.read_csv(filename)
        results.append({
            'filename': filename,
            'number_of_rows': len(df),
            'cli': df['cli'].mean()
        })

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(os.path.join(directory, f'{cli_level}_results.csv'), index=False)


train_data = pd.read_csv(TRAIN_DATA_PATH)
train_data.dropna(subset=['text'], inplace=True)
num_rows = train_data.shape[0]
print("The DataFrame has", num_rows, "rows.")

if RECALCULATE_DATASETS:
    cli_dataframe = assign_cli_to_examples(train_data)
    create_and_save_datasets(WORK_DIR, 5, cli_dataframe, 10000)

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support


class EmotionDataset(Dataset):
    def __init__(self, df, tokenizer, max_length):
        self.texts = df['text'].tolist()
        self.labels = df['shared_emotion_encoded'].tolist()
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        encoding = self.tokenizer(
            self.texts[index],
            return_tensors='pt',
            padding='max_length',
            truncation=True,
            max_length=self.max_length
        )
        return {
            'text': self.texts[index],
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(self.labels[index])
        }


class EmotionModel:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.tokenizer = None
        self.model = None
        self.stats = None

    def _stratify_and_subset(self, df, test_size=0.05):
        """Stratified splitting and optional subsetting for testing"""
        self.label_encoder = LabelEncoder()
        df['shared_emotion_encoded'] = self.label_encoder.fit_transform(df['shared_emotion'])  # Encode labels

        train_df, test_df = train_test_split(
            df, test_size=test_size, stratify=df['shared_emotion']
        )

        return train_df, test_df

    def _evaluate(self, eval_loader):
        self.model.eval()
        eval_loss = 0.0
        all_predictions = []
        all_labels = []

        with torch.no_grad():
            for batch in eval_loader:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                eval_loss += loss.item()

                logits = outputs.logits
                predictions = logits.argmax(-1)
                all_predictions.extend(predictions.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        eval_loss = eval_loss / len(eval_loader)
        eval_accuracy = accuracy_score(all_labels, all_predictions)
        precision, recall, f1_score, _ = precision_recall_fscore_support(all_labels, all_predictions,
                                                                         average='weighted')

        return eval_loss, eval_accuracy, precision, recall, f1_score

    def train_model(self, file_path, test=False, num_classes=None, epochs=3,
                    batch_size=64, learning_rate=2e-5):
        df = pd.read_csv(file_path)

        df.dropna(subset=['text'], inplace=True)

        if test:
            df = df.sample(frac=0.05)

        train_df, eval_df = self._stratify_and_subset(df)

        if num_classes is None:
            num_classes = len(df['shared_emotion'].unique())

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
        self.model = BertForSequenceClassification.from_pretrained(
            'bert-base-uncased', num_labels=num_classes
        ).to(self.device)

        train_dataset = EmotionDataset(train_df, self.tokenizer, max_length=128)
        eval_dataset = EmotionDataset(eval_df, self.tokenizer, max_length=128)

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size)

        optimizer = AdamW(self.model.parameters(), lr=learning_rate)

        stats = {}
        for epoch in range(epochs):
            self.model.train()
            train_loss = 0.0

            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}")
            for batch in progress_bar:
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['labels'].to(self.device)

                outputs = self.model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                train_loss += loss.item()

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                progress_bar.set_postfix({'loss': loss.item()})

            eval_loss, eval_acc, eval_precision, eval_recall, eval_f1 = self._evaluate(eval_loader)
            stats['eval_loss'] = eval_loss
            stats['eval_accuracy'] = eval_acc
            stats['eval_precision'] = eval_precision
            stats['eval_recall'] = eval_recall
            stats['eval_f1_score'] = eval_f1

            self.stats = stats

        return self.model, stats

    def predict_emotion(self, text):
        encoding = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True, max_length=128
        )
        input_ids = encoding['input_ids'].to(self.device)
        attention_mask = encoding['attention_mask'].to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask)

        logits = outputs.logits
        predicted_class = logits.argmax().item()
        return self.label_encoder.inverse_transform([predicted_class])[0]


file_dict = {
    'high_cli_1.csv': os.path.join(WORK_DIR, 'high', 'high_cli_1.csv'),
    'high_cli_2.csv': os.path.join(WORK_DIR, 'high', 'high_cli_2.csv'),
    'high_cli_3.csv': os.path.join(WORK_DIR, 'high', 'high_cli_3.csv'),
    'high_cli_4.csv': os.path.join(WORK_DIR, 'high', 'high_cli_4.csv'),
    'high_cli_5.csv': os.path.join(WORK_DIR, 'high', 'high_cli_5.csv'),
    'low_cli_1.csv': os.path.join(WORK_DIR, 'low', 'low_cli_1.csv'),
    'low_cli_2.csv': os.path.join(WORK_DIR, 'low', 'low_cli_2.csv'),
    'low_cli_3.csv': os.path.join(WORK_DIR, 'low', 'low_cli_3.csv'),
    'low_cli_4.csv': os.path.join(WORK_DIR, 'low', 'low_cli_4.csv'),
    'low_cli_5.csv': os.path.join(WORK_DIR, 'low', 'low_cli_5.csv'),
    'mid_cli_1.csv': os.path.join(WORK_DIR, 'mid', 'mid_cli_1.csv'),
    'mid_cli_2.csv': os.path.join(WORK_DIR, 'mid', 'mid_cli_2.csv'),
    'mid_cli_3.csv': os.path.join(WORK_DIR, 'mid', 'mid_cli_3.csv'),
    'mid_cli_4.csv': os.path.join(WORK_DIR, 'mid', 'mid_cli_4.csv'),
    'mid_cli_5.csv': os.path.join(WORK_DIR, 'mid', 'mid_cli_5.csv'),
}

files_exist = True

for filename, file_path in file_dict.items():
    if not os.path.exists(file_path):
        files_exist = False
        print(f"{filename} does not exist at {file_path}")

print(f"All files are valid: {files_exist}")

import pickle

testing = False
all_trained_models = {}

for filename, file_path in file_dict.items():
    overall_cli = calculate_dataset_cli(file_path)
    em = EmotionModel()
    model, stats = em.train_model(file_path, test=testing, num_classes=11)
    em.stats['overall_cli'] = overall_cli
    all_trained_models[filename] = em

with open(os.path.join(WORK_DIR, 'cli_all_trained_models.pkl'), 'wb') as f:
    pickle.dump(all_trained_models, f)

SAMPLE_SIZE = 1000


def calculate_validation_accuracy(model, validation_df):
    sample_df = validation_df.sample(SAMPLE_SIZE)
    correct_predictions = 0

    for index, row in sample_df.iterrows():
        predicted_emotion = model.predict_emotion(row['text'])
        if predicted_emotion == row['shared_emotion']:
            correct_predictions += 1

    return correct_predictions / SAMPLE_SIZE


all_model_results = []

for filename, model in all_trained_models.items():
    validation_df = pd.read_csv(VALIDATION_DATA_PATH)
    validation_accuracy = calculate_validation_accuracy(model, validation_df)
    model.stats['validation_accuracy'] = validation_accuracy

    all_model_results.append({
        'filename': filename,
        **model.stats
    })

results_df = pd.DataFrame(all_model_results)
results_df.to_csv(os.path.join(WORK_DIR, 'cli_all_models_final_results.csv'), index=False)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import pandas as pd
import numpy as np
from tqdm.notebook import tqdm
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


class NN_EmotionDataset(Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return self.texts[idx], self.labels[idx]


class LSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, label_size, bidirectional=True):
        super(LSTMClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.bidirectional = bidirectional
        output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        self.hidden2label = nn.Linear(output_dim, label_size)

    def forward(self, sentence, lengths):
        embeds = self.word_embeddings(sentence)
        packed_input = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        if self.bidirectional:
            lstm_out = torch.cat((lstm_out[:, -1, :self.hidden_dim], lstm_out[:, 0, self.hidden_dim:]), dim=1)
        else:
            lstm_out = lstm_out[:, -1, :]
        tag_space = self.hidden2label(lstm_out)
        tag_scores = torch.log_softmax(tag_space, dim=1)
        return tag_scores


class NN_EmotionModel:
    def __init__(self):
        self.label_encoder = None
        self.model = None
        self.max_seq_length = 0
        self.vocab_size = 0

    def train_model(self, path, tag, test=False, embedding_dim=64, hidden_dim=32, epochs=10, batch_size=1024,
                    learning_rate=0.001):
        df = pd.read_csv(path)
        df.dropna(subset=['text'], inplace=True)
        if test:
            df, _ = train_test_split(df, test_size=0.95, stratify=df["shared_emotion"])
            epochs = 2

        X_train, X_test, y_train, y_test = train_test_split(df['text'], df['shared_emotion'], test_size=0.2,
                                                            stratify=df['shared_emotion'])

        self.label_encoder = LabelEncoder()
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        y_test_encoded = self.label_encoder.transform(y_test)

        vocab = set(" ".join(X_train).split())
        self.vocab_size = len(vocab) + 2
        word_to_ix = {word: i + 1 for i, word in enumerate(vocab)}
        self.word_to_ix = word_to_ix

        def prepare_sequence(seq, to_ix):
            idxs = [to_ix.get(w, 0) for w in seq.split()]
            return torch.tensor(idxs, dtype=torch.long)

        X_train_prepared = [prepare_sequence(s, word_to_ix) for s in X_train]
        X_test_prepared = [prepare_sequence(s, word_to_ix) for s in X_test]

        X_train_prepared = [s for s in X_train_prepared if len(s) > 0]
        X_test_prepared = [s for s in X_test_prepared if len(s) > 0]

        self.max_seq_length = max(max([x.size(0) for x in X_train_prepared]), max([x.size(0) for x in X_test_prepared]))

        def pad_features(sequences, seq_len):
            features = np.zeros((len(sequences), seq_len), dtype=int)
            for i, row in enumerate(sequences):
                features[i, -len(row):] = np.array(row)[:seq_len]
            return features

        X_train_pad = pad_features(X_train_prepared, self.max_seq_length)
        X_test_pad = pad_features(X_test_prepared, self.max_seq_length)

        train_data = NN_EmotionDataset(torch.tensor(X_train_pad), torch.tensor(y_train_encoded))
        test_data = NN_EmotionDataset(torch.tensor(X_test_pad), torch.tensor(y_test_encoded))

        train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
        test_loader = DataLoader(test_data, batch_size=batch_size)

        self.model = LSTMClassifier(embedding_dim, hidden_dim, self.vocab_size, len(self.label_encoder.classes_))
        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        for epoch in range(epochs):
            total_loss = 0
            for sentence, labels in tqdm(train_loader, desc=f'Epoch {epoch + 1}/{epochs}', leave=False):
                sentence_lengths = torch.clamp(sentence.ne(0).sum(1), max=self.max_seq_length)
                self.model.zero_grad()
                tag_scores = self.model(sentence, sentence_lengths)
                loss = loss_function(tag_scores, labels)
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            print(f"Epoch {epoch + 1}, Loss: {total_loss / len(train_loader)}")

        self.model.eval()
        all_preds = []
        all_labels = []
        with torch.no_grad():
            for sentence, labels in test_loader:
                sentence_lengths = torch.clamp(sentence.ne(0).sum(1), min=1, max=self.max_seq_length)
                tag_scores = self.model(sentence, sentence_lengths)
                _, predicted = torch.max(tag_scores, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds, average='weighted')
        recall = recall_score(all_labels, all_preds, average='weighted')
        f1 = f1_score(all_labels, all_preds, average='weighted')

        self.stats = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def predict_emotion(self, text):
        if not self.model:
            raise Exception("Model not trained yet.")
        self.model.eval()
        with torch.no_grad():
            inputs = [self.word_to_ix.get(word, 0) for word in text.split()]
            inputs = [min(ix, self.vocab_size - 1) for ix in inputs]
            inputs = torch.tensor(inputs).unsqueeze(0)
            sentence_lengths = torch.clamp(inputs.ne(0).sum(1), min=1, max=self.max_seq_length)
            tag_scores = self.model(inputs, sentence_lengths)
            _, predicted = torch.max(tag_scores, 1)
            return self.label_encoder.inverse_transform(predicted.numpy())[0]


nn_file_dict = {
    'high_cli_1.csv': os.path.join(WORK_DIR, 'high', 'high_cli_1.csv'),
    'high_cli_2.csv': os.path.join(WORK_DIR, 'high', 'high_cli_2.csv'),
    'high_cli_3.csv': os.path.join(WORK_DIR, 'high', 'high_cli_3.csv'),
    'high_cli_4.csv': os.path.join(WORK_DIR, 'high', 'high_cli_4.csv'),
    'high_cli_5.csv': os.path.join(WORK_DIR, 'high', 'high_cli_5.csv'),
    'mid_cli_1.csv': os.path.join(WORK_DIR, 'mid', 'mid_cli_1.csv'),
    'mid_cli_2.csv': os.path.join(WORK_DIR, 'mid', 'mid_cli_2.csv'),
    'mid_cli_3.csv': os.path.join(WORK_DIR, 'mid', 'mid_cli_3.csv'),
    'mid_cli_4.csv': os.path.join(WORK_DIR, 'mid', 'mid_cli_4.csv'),
    'mid_cli_5.csv': os.path.join(WORK_DIR, 'mid', 'mid_cli_5.csv'),
    'low_cli_1.csv': os.path.join(WORK_DIR, 'low', 'low_cli_1.csv'),
    'low_cli_2.csv': os.path.join(WORK_DIR, 'low', 'low_cli_2.csv'),
    'low_cli_3.csv': os.path.join(WORK_DIR, 'low', 'low_cli_3.csv'),
    'low_cli_4.csv': os.path.join(WORK_DIR, 'low', 'low_cli_4.csv'),
    'low_cli_5.csv': os.path.join(WORK_DIR, 'low', 'low_cli_5.csv'),
}

files_exist = True

for filename, file_path in nn_file_dict.items():
    if not os.path.exists(file_path):
        files_exist = False
        print(f"{filename} does not exist at {file_path}")

print(f"All files are valid: {files_exist}")

import pickle

testing = False
all_nn_trained_models = {}

for filename, file_path in nn_file_dict.items():
    print(f"Processing {filename}")
    current_data = pd.read_csv(file_path)
    overall_cli = calculate_dataset_cli(file_path)
    em = NN_EmotionModel()
    em.train_model(file_path, tag='', test=testing)
    em.stats['overall_cli'] = overall_cli
    all_nn_trained_models[filename] = em

with open(os.path.join(WORK_DIR, 'BLSTM_d_cli_all_trained_models.pkl'), 'wb') as f:
    pickle.dump(all_nn_trained_models, f)

import pickle
import os

model_path = os.path.join(WORK_DIR, 'BLSTM_d_cli_all_trained_models.pkl')

with open(model_path, 'rb') as f:
    all_nn_trained_models = pickle.load(f)

SAMPLE_SIZE = 1000


def calculate_validation_accuracy(model, validation_df):
    sample_df = validation_df.groupby('shared_emotion').apply(lambda x: x.sample(frac=SAMPLE_SIZE / len(validation_df)))
    sample_df = sample_df.reset_index(drop=True)
    correct_predictions = 0

    for index, row in tqdm(sample_df.iterrows(), total=SAMPLE_SIZE):
        predicted_emotion = model.predict_emotion(row['text'])
        if predicted_emotion == row['shared_emotion']:
            correct_predictions += 1

    return correct_predictions / SAMPLE_SIZE


all_nn_model_results = []
validation_df = pd.read_csv(VALIDATION_DATA_PATH)

for filename, model in all_nn_trained_models.items():
    print(f"Evaluating for: {filename}")
    validation_accuracy = calculate_validation_accuracy(model, validation_df)
    model.stats['validation_accuracy'] = validation_accuracy

    all_nn_model_results.append({
        'filename': filename,
        **model.stats
    })

results_df = pd.DataFrame(all_nn_model_results)
results_df.to_csv(os.path.join(WORK_DIR, 'BLSTM_d_cli_all_models_final_results.csv'), index=False)
