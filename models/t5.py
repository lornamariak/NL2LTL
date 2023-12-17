"""
module to train a T5 model 
"""

import pandas as pd
from transformers import T5Tokenizer, T5ForConditionalGeneration, AdamW
from torch.utils.data import Dataset, DataLoader
import torch
from tqdm import tqdm

# Define constants for dataset URLs
TRAIN_URL = ""
TEST_URL = ""

# Load datasets
train_set = pd.read_csv(TRAIN_URL)
test_set = pd.read_csv(TEST_URL)

# Initialize T5 tokenizer and model
MODEL_NAME = "t5-base"  # You can choose the appropriate T5 model
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
model = model.to('cuda' if torch.cuda.is_available() else 'cpu')

class CustomDataset(Dataset):
    """
    Custom Dataset class for handling tokenization and data preparation
    for the T5 model.
    """

    def __init__(self, dataframe, tokenizer, max_token_len=512):
        self.tokenizer = tokenizer
        self.data = dataframe
        self.max_token_len = max_token_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data.iloc[index]
        combined_input = f"Given this natural language text {row['nl']}"

        encoding = self.tokenizer(
            combined_input,
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )
        target_encoding = self.tokenizer(
            row['formula'],
            max_length=self.max_token_len,
            padding='max_length',
            truncation=True,
            return_tensors="pt"
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': target_encoding['input_ids'].flatten()
        }

train_dataset = CustomDataset(train_set, tokenizer)
val_dataset = CustomDataset(test_set, tokenizer)

train_loader = DataLoader(train_dataset, batch_size=12, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=12)

optimizer = AdamW(model.parameters(), lr=1e-5)

# Set number of epochs
NUM_EPOCHS = 50

def evaluate(model, val_loader, tokenizer):
    """
    Function to evaluate the model on the validation set.
    """
    model.eval()
    total_eval_loss = 0
    total_matches = 0
    total_examples = 0

    with torch.no_grad():
        for batch in val_loader:
            input_ids = batch['input_ids'].to(model.device)
            attention_mask = batch['attention_mask'].to(model.device)
            labels = batch['labels'].to(model.device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            output_token_ids = torch.argmax(outputs.logits, dim=-1)

            labels_list = labels.tolist()

            for i, output_id in enumerate(output_token_ids):
                output_decoded = tokenizer.decode(output_id, skip_special_tokens=True)
                label_decoded = tokenizer.decode(labels_list[i], skip_special_tokens=True)
                
                if output_decoded == label_decoded:
                    total_matches += 1

                total_examples += 1

            loss = outputs.loss
            total_eval_loss += loss.item()

    avg_val_loss = total_eval_loss / len(val_loader)
    accuracy = total_matches / total_examples
    return avg_val_loss, accuracy


# Training loop
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    for batch in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{NUM_EPOCHS}", unit="batch"):
        optimizer.zero_grad()
        input_ids = batch['input_ids'].to(model.device)
        attention_mask = batch['attention_mask'].to(model.device)
        labels = batch['labels'].to(model.device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        total_train_loss += loss.item()

        loss.backward()
        optimizer.step()

    avg_train_loss = total_train_loss / len(train_loader)
    eval_loss, acc = evaluate(model, val_loader, tokenizer)
    print(f"Training Loss: {avg_train_loss}, Eval loss: {eval_loss}, Eval Acc: {acc}")
