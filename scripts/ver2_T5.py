import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments


# pip install transformers sentencepiece torch scikit-learn pandas


df = pd.read_csv('Baza customer Telecom v2.csv')
df.columns = df.columns.str.strip()



df = df.drop(columns=['PID'])
df['Suspended_subscribers'] = df['Suspended_subscribers'].fillna(0)
df['Not_Active_subscribers'] = df['Not_Active_subscribers'].fillna(0)
df['ARPU'] = df['ARPU'].fillna(df['ARPU'].median())
target_col = 'CHURN'
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
numeric_cols = [c for c in numeric_cols if c not in [target_col, 'Suspended_subscribers', 'Not_Active_subscribers', 'ARPU']]
df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
categorical_cols = [c for c in df.columns if df[c].dtype == 'object' and c != target_col]
df[categorical_cols] = df[categorical_cols].fillna('Unknown')


def make_input(row):
    parts = [f"{f}: {row[f]}" for f in df.columns if f != target_col]
    return " | ".join(parts)
df['text_input'] = df.apply(make_input, axis=1)
df['text_label'] = df[target_col].map({'No': 'no', 'Yes': 'yes'})


X_temp, X_test_df, y_temp, y_test = train_test_split(
    df, df['text_label'], test_size=0.2, random_state=42, stratify=df['text_label']
)
X_train_df, X_val_df, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp
)


tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')


class TelecomDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.inputs = tokenizer(
            df['text_input'].tolist(), max_length=512, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        label_enc = tokenizer(
            df['text_label'].tolist(), max_length=4, truncation=True,
            padding='max_length', return_tensors='pt'
        )
        self.labels = label_enc.input_ids
        self.labels[self.labels == tokenizer.pad_token_id] = -100

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item

train_dataset = TelecomDataset(X_train_df)
val_dataset   = TelecomDataset(X_val_df)
test_dataset  = TelecomDataset(X_test_df)


training_args = TrainingArguments(
    output_dir='out',
    do_train=True,
    do_eval=True,
    per_device_train_batch_size=8,
    num_train_epochs=3,
    save_total_limit=1
)
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset
)
trainer.train()


from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)


test_labels = [1 if lbl == 'yes' else 0 for lbl in X_test_df['text_label'].tolist()]


test_loader = DataLoader(test_dataset, batch_size=8)
all_preds = []
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    outputs = model.generate(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_length=2
    )
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    preds = [1 if text.strip().lower() == 'yes' else 0 for text in decoded]
    all_preds.extend(preds)


assert len(test_labels) == len(all_preds), f"Mismatch: {len(test_labels)} vs {len(all_preds)}"


print('Accuracy:', accuracy_score(test_labels, all_preds))
print('Classification Report:')
print(classification_report(test_labels, all_preds, zero_division=0))
print('Confusion Matrix:')
print(confusion_matrix(test_labels, all_preds))

from sklearn.metrics import precision_score, recall_score, f1_score

print("Precision (yes):", precision_score(test_labels, all_preds, pos_label=1))
print("Recall    (yes):", recall_score(   test_labels, all_preds, pos_label=1))
print("F1-score  (yes):",    f1_score(    test_labels, all_preds, pos_label=1))
