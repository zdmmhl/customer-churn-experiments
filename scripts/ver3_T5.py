import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from transformers import T5Tokenizer, T5ForConditionalGeneration, Trainer, TrainingArguments
from torch.utils.data import DataLoader


df = pd.read_csv('processed_customer_data_plus_improved.csv')
df['CHURN'] = df['CHURN'].astype(int)


features = [col for col in df.columns if col != 'CHURN']
def make_input(row):
    parts = [f"{f}: {row[f]}" for f in features]
    return " | ".join(parts)

df['text_input'] = df.apply(make_input, axis=1)
df['text_label'] = df['CHURN'].map({0: "no", 1: "yes"})


X_temp, X_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['text_label'])
X_train, X_val = train_test_split(X_temp, test_size=0.25, random_state=42, stratify=X_temp['text_label'])


# pos = X_train[X_train['text_label'] == 'yes']
# neg = X_train[X_train['text_label'] == 'no']
# repeat_factor = len(neg) // len(pos) - 1
# X_train = pd.concat([X_train] + [pos] * repeat_factor, ignore_index=True).sample(frac=1, random_state=42)


tokenizer = T5Tokenizer.from_pretrained('t5-small')
model = T5ForConditionalGeneration.from_pretrained('t5-small')


class TelecomDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.inputs = tokenizer(df['text_input'].tolist(), max_length=512, truncation=True, padding='max_length', return_tensors='pt')
        labels = tokenizer(df['text_label'].tolist(), max_length=4, truncation=True, padding='max_length', return_tensors='pt')
        labels.input_ids[labels.input_ids == tokenizer.pad_token_id] = -100
        self.labels = labels.input_ids

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx] for key, val in self.inputs.items()}
        item['labels'] = self.labels[idx]
        return item


train_dataset = TelecomDataset(X_train)
val_dataset = TelecomDataset(X_val)
test_dataset = TelecomDataset(X_test)


training_args = TrainingArguments(
    output_dir="out_t5_plus",
    do_train=True,
    do_eval=False,
    logging_steps=50,
    save_total_limit=1,
    per_device_train_batch_size=8,
    num_train_epochs=3
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    tokenizer=tokenizer
)


trainer.train()


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
test_loader = DataLoader(test_dataset, batch_size=8)

all_preds = []
all_labels = X_test['text_label'].map({'no': 0, 'yes': 1}).tolist()
for batch in test_loader:
    input_ids = batch['input_ids'].to(device)
    attention_mask = batch['attention_mask'].to(device)
    outputs = model.generate(input_ids=input_ids, attention_mask=attention_mask, max_length=2)
    decoded = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    preds = [1 if text.strip().lower() == 'yes' else 0 for text in decoded]
    all_preds.extend(preds)


print("Accuracy:", accuracy_score(all_labels, all_preds))
print("Classification Report:")
print(classification_report(all_labels, all_preds, zero_division=0))
print("Confusion Matrix:")
print(confusion_matrix(all_labels, all_preds))
