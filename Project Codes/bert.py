import pandas as pd
import re
import string
from nltk.corpus import stopwords
from nltk.stem import SnowballStemmer
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertTokenizer, BertForSequenceClassification
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# Load dataset
df = pd.read_csv("spam.csv", encoding="latin-1")
# Drop columns with NaN values
df.dropna(axis=1, inplace=True)
# Rename columns
df.columns = ['label', 'message']

def clean_text(text):
    # Make text lowercase, remove text in square brackets, remove links, HTML tags, punctuation, newlines, extra spaces, and words containing numbers
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f'[{re.escape(string.punctuation)}]', '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

# Define stopwords
stop_words = set(stopwords.words('english'))
stop_words.update({'u', 'im', 'c'})

# Initialize stemmer
stemmer = SnowballStemmer("english")

def preprocess_data(text):
    # Clean text, remove stopwords, and apply stemming
    text = clean_text(text)
    words = text.split()
    filtered_words = [word for word in words if word not in stop_words]
    stemmed_words = [stemmer.stem(word) for word in filtered_words]
    return ' '.join(stemmed_words)

# Apply preprocessing to messages
df['message_preprocessed'] = df['message'].apply(preprocess_data)

# Encode labels as integers
label_encoder = LabelEncoder()
df['label_encoded'] = label_encoder.fit_transform(df['label'])


class SpamDataset(Dataset):
    def __init__(self, messages, labels, tokenizer, max_len):
        self.messages = messages
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.messages)

    def __getitem__(self, item):
        # Get a sample at the given index
        message = str(self.messages[item])
        label = self.labels[item]

        # Tokenize the message
        encoding = self.tokenizer.encode_plus(
            message,
            add_special_tokens=True,  # Add [CLS] and [SEP] tokens
            max_length=self.max_len,  # Set maximum sequence length
            return_token_type_ids=False,  # No token type IDs needed
            padding='max_length',  # Pad sequences to max length
            return_attention_mask=True,  # Return attention mask
            return_tensors='pt',  # Return tensors in PyTorch format
        )

        return {
            'message_text': message,
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }

# Split the data into training and validation sets (80% train, 20% val)
df_train, df_val = train_test_split(df, test_size=0.2, random_state=42)

# Initialize the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# Create the training dataset
train_dataset = SpamDataset(
    messages=df_train['message_preprocessed'].values,
    labels=df_train['label_encoded'].values,
    tokenizer=tokenizer,
    max_len=128
)

# Create the validation dataset
val_dataset = SpamDataset(
    messages=df_val['message_preprocessed'].values,
    labels=df_val['label_encoded'].values,
    tokenizer=tokenizer,
    max_len=128
)

# Create DataLoaders for training and validation datasets
train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16)

# Initialize the BERT model for sequence classification
model = BertForSequenceClassification.from_pretrained(
    'bert-base-uncased',        # Use the base BERT model (uncased)
    num_labels=2,               # Number of classes: spam and ham
    output_attentions=False,    # Do not return attention weights
    output_hidden_states=False  # Do not return hidden states
)

# Set device to MPS if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# Move model to the selected device
model = model.to(device)

# Define the optimizer
optimizer = AdamW(model.parameters(), lr=1e-5)

# Calculate total training steps (batches per epoch * 4 epochs)
total_steps = len(train_loader) * 3

# Set up the learning rate scheduler
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=0,
    num_training_steps=total_steps
)

def train_epoch(model, data_loader, optimizer, device, scheduler):
    # Set model to training mode
    model.train()
    total_loss = 0
    correct_predictions = 0
    progress = tqdm(data_loader, desc='Training', leave=False)

    for data in progress:
        # Move data to the specified device
        input_ids = data['input_ids'].to(device)
        attention_mask = data['attention_mask'].to(device)
        labels = data['labels'].to(device)

        # Forward pass
        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
        loss = outputs.loss
        logits = outputs.logits
        _, preds = torch.max(logits, dim=1)

        # Track correct predictions and loss
        correct_predictions += torch.sum(preds == labels)
        total_loss += loss.item()

        # Backward pass and optimization step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # Update progress bar with current loss
        progress.set_postfix({'loss': loss.item()})

    # Return accuracy and average loss
    return correct_predictions.float() / len(data_loader.dataset), total_loss / len(data_loader)

def eval_model(model, data_loader, device):
    # Set model to evaluation mode
    model.eval()
    total_loss = 0
    correct_predictions = 0
    progress = tqdm(data_loader, desc='Validating', leave=False)

    with torch.no_grad():
        for data in progress:
            # Move data to the specified device
            input_ids = data['input_ids'].to(device)
            attention_mask = data['attention_mask'].to(device)
            labels = data['labels'].to(device)

            # Forward pass
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs.loss
            logits = outputs.logits
            _, preds = torch.max(logits, dim=1)

            # Track correct predictions and loss
            correct_predictions += torch.sum(preds == labels)
            total_loss += loss.item()

            # Update progress bar with current validation loss
            progress.set_postfix({'val_loss': loss.item()})

    # Return accuracy and average loss
    return correct_predictions.float() / len(data_loader.dataset), total_loss / len(data_loader)

# Set the number of epochs
epochs = 3

# Lists to store metrics for plotting
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

# Training loop
for epoch in range(epochs):
    print(f'Epoch {epoch + 1}/{epochs}')
    print('-' * 10)

    # Training phase
    train_acc, train_loss = train_epoch(
        model,
        train_loader,
        optimizer,
        device,
        scheduler
    )
    train_losses.append(train_loss)
    train_accuracies.append(train_acc.item())

    print(f'Train loss: {train_loss:.4f}, accuracy: {train_acc:.4f}')

    # Validation phase
    val_acc, val_loss = eval_model(
        model,
        val_loader,
        device
    )
    val_losses.append(val_loss)
    val_accuracies.append(val_acc.item())

    print(f'Val loss: {val_loss:.4f}, accuracy: {val_acc:.4f}')
    print()

# Print training summary
print("Training Complete")
print(f"Final Train Loss: {train_losses[-1]:.4f}, Train Accuracy: {train_accuracies[-1]:.4f}")
print(f"Final Val Loss: {val_losses[-1]:.4f}, Val Accuracy: {val_accuracies[-1]:.4f}")