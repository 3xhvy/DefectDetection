import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
import logging
import re
from collections import Counter

logger = logging.getLogger(__name__)

# Text preprocessing
def preprocess_code(code):
    """Clean and tokenize code"""
    # Replace tabs and newlines with spaces
    code = re.sub(r'[\t\n]', ' ', code)
    # Remove extra spaces
    code = re.sub(r'\s+', ' ', code)
    # Split by spaces and special characters, but keep the special characters
    tokens = []
    current_token = ""
    for char in code:
        if char.isalnum() or char == '_':
            current_token += char
        else:
            if current_token:
                tokens.append(current_token)
                current_token = ""
            if not char.isspace():
                tokens.append(char)
    if current_token:
        tokens.append(current_token)
    return tokens

class SimpleVocab:
    """A simple vocabulary class to replace torchtext.vocab"""
    def __init__(self, tokens, min_freq=2):
        self.token_to_idx = {'<unk>': 0, '<pad>': 1}
        self.idx_to_token = ['<unk>', '<pad>']
        
        # Count token frequencies
        counter = Counter(tokens)
        
        # Add tokens that meet minimum frequency
        for token, count in counter.items():
            if count >= min_freq:
                self.token_to_idx[token] = len(self.idx_to_token)
                self.idx_to_token.append(token)
    
    def __getitem__(self, token):
        return self.token_to_idx.get(token, 0)  # Return <unk> index if token not found
    
    def __len__(self):
        return len(self.idx_to_token)

def get_vocab(code_samples, min_freq=2):
    """Build vocabulary from code samples"""
    all_tokens = []
    for code in code_samples:
        all_tokens.extend(preprocess_code(code))
    
    return SimpleVocab(all_tokens, min_freq)

def text_to_indices(text, vocab, max_length=512):
    """Convert text to indices using vocabulary"""
    tokens = preprocess_code(text)
    indices = [vocab[token] for token in tokens]
    
    # Truncate or pad to max_length
    if len(indices) > max_length:
        indices = indices[:max_length]
    else:
        indices = indices + [1] * (max_length - len(indices))  # 1 is <pad>
    
    return indices

class CodeTextDataset(Dataset):
    """Dataset for code as text"""
    def __init__(self, texts, labels, vocab, max_length=512):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.max_length = max_length
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        
        # Convert text to indices
        indices = text_to_indices(text, self.vocab, self.max_length)
        
        return torch.tensor(indices, dtype=torch.long), torch.tensor(label, dtype=torch.float)

def prepare_text_data(train_texts, train_labels, val_texts, val_labels, test_texts, test_labels, batch_size=32, max_length=512):
    """Prepare data for sequence models"""
    logger.info("Building vocabulary...")
    vocab = get_vocab(train_texts)
    logger.info(f"Vocabulary size: {len(vocab)}")
    
    logger.info("Creating datasets...")
    train_dataset = CodeTextDataset(train_texts, train_labels, vocab, max_length)
    val_dataset = CodeTextDataset(val_texts, val_labels, vocab, max_length)
    test_dataset = CodeTextDataset(test_texts, test_labels, vocab, max_length)
    
    logger.info("Creating dataloaders...")
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_texts, val_texts, test_texts, vocab, train_dataloader, val_dataloader, test_dataloader

class BiLSTMModel(nn.Module):
    """Bidirectional LSTM model for code vulnerability detection"""
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)  # 1 is the index for <pad>
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers=n_layers, 
                           bidirectional=True, dropout=dropout, batch_first=True)
        self.fc1 = nn.Linear(hidden_dim * 2, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch size, seq len]
        embedded = self.embedding(text)
        # embedded shape: [batch size, seq len, embedding dim]
        
        output, (hidden, cell) = self.lstm(embedded)
        # output shape: [batch size, seq len, hidden dim * 2]
        # hidden shape: [n layers * 2, batch size, hidden dim]
        
        # Concatenate the final forward and backward hidden states
        hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
        # hidden shape: [batch size, hidden dim * 2]
        
        hidden = self.dropout(hidden)
        hidden = F.relu(self.fc1(hidden))
        hidden = self.dropout(hidden)
        
        return self.fc2(hidden)

class TransformerModel(nn.Module):
    """Transformer model for code vulnerability detection"""
    def __init__(self, vocab_size, embedding_dim, num_heads, hidden_dim, num_layers, output_dim, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=1)  # 1 is the index for <pad>
        self.pos_encoder = PositionalEncoding(embedding_dim, dropout)
        
        encoder_layers = nn.TransformerEncoderLayer(
            d_model=embedding_dim, 
            nhead=num_heads, 
            dim_feedforward=hidden_dim, 
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layers, num_layers)
        
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, text):
        # text shape: [batch size, seq len]
        
        # Create mask for padding tokens
        mask = (text == 1).to(text.device)  # 1 is the index for <pad>
        
        embedded = self.embedding(text) * np.sqrt(self.embedding.embedding_dim)
        # embedded shape: [batch size, seq len, embedding dim]
        
        embedded = self.pos_encoder(embedded)
        
        # Apply transformer with padding mask
        output = self.transformer_encoder(embedded, src_key_padding_mask=mask)
        # output shape: [batch size, seq len, embedding dim]
        
        # Global average pooling
        output = output.mean(dim=1)
        # output shape: [batch size, embedding dim]
        
        output = self.dropout(output)
        output = F.relu(self.fc1(output))
        output = self.dropout(output)
        
        return self.fc2(output)

class PositionalEncoding(nn.Module):
    """Positional encoding for transformer model"""
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        # x shape: [batch size, seq len, embedding dim]
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)
