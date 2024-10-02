import os
import pandas as pd
import re
import string
from sklearn.model_selection import train_test_split
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset
import torch

# Set the device to GPU if available, otherwise CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load datasets
df_fake = pd.read_csv('dataset/FakeNews.csv')
df_true = pd.read_csv('dataset/TrueNews.csv')

# Assign class labels
df_fake["class"] = 0
df_true["class"] = 1

# Drop last 10 rows for manual testing
df_fake.drop(df_fake.tail(10).index, inplace=True)
df_true.drop(df_true.tail(10).index, inplace=True)

# Merge datasets
df_merge = pd.concat([df_fake, df_true], axis=0).reset_index(drop=True)

# Shuffle the dataset
df = df_merge.sample(frac=1).reset_index(drop=True)

# Text preprocessing
def remove_punctuation(text):
    return re.sub(f"[{string.punctuation}]", "", text)

df['cleaned_text'] = df['short_description'].apply(remove_punctuation)

# Define custom Hindi stemmer
suffixes = {
    1: ["ो", "े", "ू", "ु", "ी", "ि", "ा"],
    2: ["कर", "ाओ", "िए", "ाई", "ाए", "ने", "नी", "ना", "ते", "ीं", "ती", "ता", "ाँ", "ां", "ों", "ें"],
    3: ["ाकर", "ाइए", "ाईं", "ाया", "ेगी", "ेगा", "ोगी", "ोगे", "ाने", "ाना", "ाते", "ाती", "ाता", "तीं", "ाओं", "ाएं", "ुओं", "ुएं", "ुआं"],
    4: ["ाएगी", "ाएगा", "ाओगी", "ाओगे", "एंगी", "ेंगी", "एंगे", "ेंगे", "ूंगी", "ूंगा", "ातीं", "नाओं", "नाएं", "ताओं", "ताएं", "ियाँ", "ियों", "ियां"],
    5: ["ाएंगी", "ाएंगे", "ाऊंगी", "ाऊंगा", "ाइयाँ", "ाइयों", "ाइयां"],
}

def custom_hindi_stemmer(word):
    for L in range(5, 0, -1):
        if L in suffixes:
            for suffix in suffixes[L]:
                if word.endswith(suffix):
                    return word[:-L]
    return word

def stem_text(text):
    tokens = text.split()
    stemmed_tokens = [custom_hindi_stemmer(token) for token in tokens]
    return ' '.join(stemmed_tokens)

df['stemmed_text'] = df['cleaned_text'].apply(stem_text)

# Remove custom stopwords
custom_stopwords = ['मैं', 'मुझको', 'मेरा', ...]

def remove_stopwords(text):
    tokens = text.split()
    filtered_tokens = [word for word in tokens if word not in custom_stopwords]
    return ' '.join(filtered_tokens)

df['final_text'] = df['stemmed_text'].apply(remove_stopwords)

# Keep only relevant columns
df = df[['class', 'final_text']]

# Rename class column
df = df.rename(columns={'class': 'label'})

# Define paths
model_dir = './model'
tokenizer_dir = './saved_tokenizer'

# Tokenizer initialization
tokenizer = DistilBertTokenizer.from_pretrained(tokenizer_dir if os.path.exists(tokenizer_dir) else "distilbert-base-uncased")

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)



# Tokenization function with padding
def tokenize(batch):
    return tokenizer(batch['final_text'], padding='max_length', truncation=True, max_length=128)  # Adjust max_length as needed

# Convert DataFrame to Hugging Face Dataset
dataset = Dataset.from_pandas(df)

# Apply tokenization to the dataset
dataset = dataset.map(tokenize, batched=True)
dataset.set_format('torch', columns=['input_ids', 'attention_mask', 'label'])

# Split dataset
train_dataset, test_dataset = dataset.train_test_split(test_size=0.2).values()




# Check if the model exists
if os.path.exists(model_dir):
    print("Loading the saved model...")
    model = DistilBertForSequenceClassification.from_pretrained(model_dir).to(device)
else:
    print("Training the model from scratch...")
    model = DistilBertForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir=model_dir,
        num_train_epochs=1,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        warmup_steps=500,
        weight_decay=0.01,
        evaluation_strategy="epoch",
        logging_dir='./logs',
        save_steps=1000,
        save_total_limit=2
    )

    # Trainer initialization
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=test_dataset
    )
    
    # Train the model
    trainer.train()

    # Save model and tokenizer
    model.save_pretrained(model_dir)
    tokenizer.save_pretrained(tokenizer_dir)

# Ensure trainer is initialized regardless of training or loading
training_args = TrainingArguments(
    output_dir=model_dir,
    num_train_epochs=1,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    warmup_steps=500,
    weight_decay=0.01,
    evaluation_strategy="epoch",
    logging_dir='./logs',
    save_steps=1000,
    save_total_limit=2
)

# Initialize trainer for evaluation
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset
)

# Evaluate the model
results = trainer.evaluate()
print(results)

# Testing the model with user input
def test_model_on_input():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    user_input = input("Enter a Hindi news headline: ")

    # Tokenize the user input
    encoded_input = tokenizer(user_input, return_tensors='pt', padding=True, truncation=True).to(device)

    # Move the model to the device
    model.to(device)

    # Make the prediction
    with torch.no_grad():
        output = model(**encoded_input)
        prediction = torch.argmax(output.logits, dim=-1).item()

    # Interpret the prediction
    result = "Fake" if prediction == 0 else "Real"
    print(f"The model predicts this news to be: {result}")

# Call the function to test
test_model_on_input()