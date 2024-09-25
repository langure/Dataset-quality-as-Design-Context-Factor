import os

import pandas as pd

# GLOBAL VARIABLES
SOURCE_DATA_PATH = './workdir'
TRAIN_DATA_PATH = os.path.join(SOURCE_DATA_PATH, 'train_dataframe.csv')
VALIDATION_DATA_PATH = os.path.join(SOURCE_DATA_PATH, 'validation_dataframe.csv')

RECALCULATE_DATASETS = False

WORK_DIR = './workdir/gini'
BALANCED_GINI_DATAFRAME = os.path.join(WORK_DIR, 'balanced_gini.csv')
MEDIUM_GINI_DATAFRAME = os.path.join(WORK_DIR, 'medium_gini.csv')
UNBALANCED_GINI_DATAFRAME = os.path.join(WORK_DIR, 'unbalanced_gini.csv')

VALIDATION_FRACTION = 0.15
BALANCED_GINI_INDEX = 0
MEDIUM_GINI_INDEX = 0.4
UNBALANCED_GINI_INDEX = 0.6
TOLERANCE = 0.05

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


def gini_coefficient(df):
    label_counts = df['shared_emotion'].value_counts().sort_values()
    array = np.array(label_counts)
    array = array.flatten()

    if np.amin(array) < 0:
        array -= np.amin(array)
    array = array + 0.0000001
    array = np.sort(array)
    index = np.arange(1, array.shape[0] + 1)
    n = array.shape[0]
    gini_index = ((np.sum((2 * index - n - 1) * array)) / (n * np.sum(array)))

    return gini_index


def redistribute_gini_index(df, target_index, change_step, tolerance, max_iterations):
    try:
        label_counts = df['shared_emotion'].value_counts()
        min_label_count = label_counts.min()
        balanced_df = pd.concat(
            [df[df['shared_emotion'] == label].sample(min_label_count) for label in label_counts.index])

        b_label_counts = balanced_df['shared_emotion'].value_counts()
        current_index = gini_coefficient(balanced_df)

        if abs(current_index - target_index) < tolerance:
            return balanced_df, current_index

        original_data_minus_balanced = df.drop(balanced_df.index)

        label_order = list(label_counts.index)

        ptr = 1
        progress_bar = tqdm(total=max_iterations, desc=f"Gini Index: {current_index:.4f}")
        for iteration in range(max_iterations):
            accumulator = ptr
            for label in label_order:
                if accumulator == 0:
                    break
                remove_count = min(label_counts[label], change_step)
                to_remove = balanced_df[balanced_df['shared_emotion'] == label].head(remove_count).index
                balanced_df = balanced_df.drop(to_remove)
                accumulator -= 1
            three_most_frequent_labels = b_label_counts.nlargest(3).index
            for label in three_most_frequent_labels:
                to_add = original_data_minus_balanced[original_data_minus_balanced['shared_emotion'] == label].head(
                    change_step)
                balanced_df = pd.concat([balanced_df, to_add])
                original_data_minus_balanced = original_data_minus_balanced.drop(to_add.index)

            b_label_counts = balanced_df['shared_emotion'].value_counts()
            current_index = gini_coefficient(balanced_df)

            if abs(current_index - target_index) < tolerance:
                progress_bar.close()
                return balanced_df, current_index

            current_diff = abs(current_index - target_index)

            progress_bar.set_description(f"Gini Index: {current_index:.4f}")
            progress_bar.update(1)
            ptr += 1

        progress_bar.close()
        return balanced_df, current_index

    except KeyError:
        print("Error: The 'shared_emotion' column does not exist in the dataframe.")
        return None, None


def create_and_save_datasets(tag, num_iterations, target_index, change_step, tolerance, max_iterations):
    results = []

    target_dir = os.path.join(WORK_DIR, tag)
    os.makedirs(target_dir, exist_ok=True)

    for iteration in range(1, num_iterations + 1):
        target_df, gini_index = redistribute_gini_index(train_data, target_index=target_index, change_step=change_step,
                                                        tolerance=tolerance, max_iterations=max_iterations)
        output_filename = os.path.join(target_dir, f"{tag}_{iteration}.csv")
        target_df.to_csv(output_filename, index=False)
        results.append({
            'file_name': output_filename,
            'number_of_rows': len(target_df),
            'gini_index': gini_index
        })
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(WORK_DIR, f'{tag}_results.csv'), index=False)


train_data = pd.read_csv(TRAIN_DATA_PATH)
original_gini_index = gini_coefficient(train_data)
print("\nOriginal Gini Index:", original_gini_index)
num_rows = train_data.shape[0]
print("The DataFrame has", num_rows, "rows.")

num_iterations = 5
if RECALCULATE_DATASETS:
    create_and_save_datasets('balanced', num_iterations, target_index=BALANCED_GINI_INDEX, change_step=20,
                             tolerance=TOLERANCE, max_iterations=1000)
num_iterations = 5
if RECALCULATE_DATASETS:
    create_and_save_datasets('medium', num_iterations, target_index=MEDIUM_GINI_INDEX, change_step=20,
                             tolerance=TOLERANCE, max_iterations=1000)
num_iterations = 5
if RECALCULATE_DATASETS:
    create_and_save_datasets('unbalanced', num_iterations, target_index=UNBALANCED_GINI_INDEX, change_step=5,
                             tolerance=TOLERANCE, max_iterations=10000)

import pandas as pd
from transformers import BertTokenizer, BertForSequenceClassification, AdamW
from torch.utils.data import Dataset
from sklearn.metrics import precision_recall_fscore_support

from tqdm import tqdm


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

        stats = {}  # Store overall stats
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
    'balanced_1.csv': os.path.join(WORK_DIR, 'balanced', 'balanced_1.csv'),
    'balanced_2.csv': os.path.join(WORK_DIR, 'balanced', 'balanced_2.csv'),
    'balanced_3.csv': os.path.join(WORK_DIR, 'balanced', 'balanced_3.csv'),
    'balanced_4.csv': os.path.join(WORK_DIR, 'balanced', 'balanced_4.csv'),
    'balanced_5.csv': os.path.join(WORK_DIR, 'balanced', 'balanced_5.csv'),
    'medium_1.csv': os.path.join(WORK_DIR, 'medium', 'medium_1.csv'),
    'medium_2.csv': os.path.join(WORK_DIR, 'medium', 'medium_2.csv'),
    'medium_3.csv': os.path.join(WORK_DIR, 'medium', 'medium_3.csv'),
    'medium_4.csv': os.path.join(WORK_DIR, 'medium', 'medium_4.csv'),
    'medium_5.csv': os.path.join(WORK_DIR, 'medium', 'medium_5.csv'),
    'unbalanced_1.csv': os.path.join(WORK_DIR, 'unbalanced', 'unbalanced_1.csv'),
    'unbalanced_2.csv': os.path.join(WORK_DIR, 'unbalanced', 'unbalanced_2.csv'),
    'unbalanced_3.csv': os.path.join(WORK_DIR, 'unbalanced', 'unbalanced_3.csv'),
    'unbalanced_4.csv': os.path.join(WORK_DIR, 'unbalanced', 'unbalanced_4.csv'),
    'unbalanced_5.csv': os.path.join(WORK_DIR, 'unbalanced', 'unbalanced_5.csv'),
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
    current_data = pd.read_csv(file_path)
    overall_gini = gini_coefficient(current_data)
    em = EmotionModel()
    model, stats = em.train_model(file_path, test=testing, num_classes=11)
    em.stats['overall_gini'] = overall_gini
    all_trained_models[filename] = em

with open(os.path.join(WORK_DIR, 'gini_all_trained_models.pkl'), 'wb') as f:
    pickle.dump(all_trained_models, f)

SAMPLE_SIZE = 340


def calculate_validation_accuracy(model, validation_df):
    sample_df = validation_df.groupby('shared_emotion').apply(lambda x: x.sample(frac=SAMPLE_SIZE / len(validation_df)))
    sample_df = sample_df.reset_index(drop=True)
    correct_predictions = 0

    for index, row in tqdm(sample_df.iterrows(), total=SAMPLE_SIZE):
        predicted_emotion = model.predict_emotion(row['text'])
        if predicted_emotion == row['shared_emotion']:
            correct_predictions += 1

    return correct_predictions / SAMPLE_SIZE


all_model_results = []
validation_df = pd.read_csv(VALIDATION_DATA_PATH)

for filename, model in all_trained_models.items():
    print(f"Evaluating for: {filename}")
    validation_accuracy = calculate_validation_accuracy(model, validation_df)
    model.stats['validation_accuracy'] = validation_accuracy

    all_model_results.append({
        'filename': filename,
        **model.stats
    })

results_df = pd.DataFrame(all_model_results)
results_df.to_csv(os.path.join(WORK_DIR, 'gini_all_models_final_results.csv'), index=False)

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
        # Multiply hidden dimension by 2 if bidirectional
        output_dim = hidden_dim * 2 if self.bidirectional else hidden_dim

        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # Set bidirectional=True for bidirectional LSTM
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        # Adjust the input dimension of the linear layer if LSTM is bidirectional
        self.hidden2label = nn.Linear(output_dim, label_size)

    def forward(self, sentence, lengths):
        embeds = self.word_embeddings(sentence)
        packed_input = pack_padded_sequence(embeds, lengths, batch_first=True, enforce_sorted=False)
        lstm_out, _ = self.lstm(packed_input)
        lstm_out, _ = pad_packed_sequence(lstm_out, batch_first=True)
        # If bidirectional, lstm_out[:, -1, :] gathers the last step of the forward direction
        # and lstm_out[:, 0, :] gathers the first step of the backward direction, we concatenate them.
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
    'balanced_1.csv': os.path.join(WORK_DIR, 'balanced', 'balanced_1.csv'),
    'balanced_2.csv': os.path.join(WORK_DIR, 'balanced', 'balanced_2.csv'),
    'balanced_3.csv': os.path.join(WORK_DIR, 'balanced', 'balanced_3.csv'),
    'balanced_4.csv': os.path.join(WORK_DIR, 'balanced', 'balanced_4.csv'),
    'balanced_5.csv': os.path.join(WORK_DIR, 'balanced', 'balanced_5.csv'),
    'medium_1.csv': os.path.join(WORK_DIR, 'medium', 'medium_1.csv'),
    'medium_2.csv': os.path.join(WORK_DIR, 'medium', 'medium_2.csv'),
    'medium_3.csv': os.path.join(WORK_DIR, 'medium', 'medium_3.csv'),
    'medium_4.csv': os.path.join(WORK_DIR, 'medium', 'medium_4.csv'),
    'medium_5.csv': os.path.join(WORK_DIR, 'medium', 'medium_5.csv'),
    'unbalanced_1.csv': os.path.join(WORK_DIR, 'unbalanced', 'unbalanced_1.csv'),
    'unbalanced_2.csv': os.path.join(WORK_DIR, 'unbalanced', 'unbalanced_2.csv'),
    'unbalanced_3.csv': os.path.join(WORK_DIR, 'unbalanced', 'unbalanced_3.csv'),
    'unbalanced_4.csv': os.path.join(WORK_DIR, 'unbalanced', 'unbalanced_4.csv'),
    'unbalanced_5.csv': os.path.join(WORK_DIR, 'unbalanced', 'unbalanced_5.csv'),
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
    overall_gini = gini_coefficient(current_data)
    em = NN_EmotionModel()
    em.train_model(file_path, tag='', test=testing)
    em.stats['overall_gini'] = overall_gini
    all_nn_trained_models[filename] = em

with open(os.path.join(WORK_DIR, 'BLSTM_gini_all_trained_models.pkl'), 'wb') as f:
    pickle.dump(all_nn_trained_models, f)

import pickle
import os

model_path = os.path.join(WORK_DIR, 'BLSTM_gini_all_trained_models.pkl')

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
results_df.to_csv(os.path.join(WORK_DIR, 'BLSTM_gini_all_models_final_results.csv'), index=False)
