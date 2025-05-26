import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from tqdm import tqdm
import math
import random
from music21 import stream, note, chord, tempo, meter, duration
import pickle

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MusicDataset(Dataset):
    def __init__(self, tokenized_dir, seq_length=512, vocab_size=None):
        self.seq_length = seq_length
        self.sequences = []
        
        # Check if directory exists
        if not os.path.exists(tokenized_dir):
            raise FileNotFoundError(f"Tokenized directory not found: {tokenized_dir}")
        
        # Load all tokenized files
        all_tokens = []
        file_count = 0
        
        print(f"Looking for tokenized files in: {tokenized_dir}")
        for root, dirs, files in os.walk(tokenized_dir):
            print(f"Checking directory: {root} with {len(files)} files")
            for file in files:
                if file.endswith('.txt'):
                    file_count += 1
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r') as f:
                            content = f.read().strip()
                            if content:  # Only process non-empty files
                                tokens = list(map(int, content.split()))
                                all_tokens.extend(tokens)
                                if file_count <= 3:  # Show first few files for debugging
                                    print(f"Loaded {len(tokens)} tokens from {file}")
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
        
        print(f"Found {file_count} tokenized files")
        
        if len(all_tokens) == 0:
            raise ValueError("No tokens found! Please check your tokenized directory and file contents.")
        
        # Build vocabulary
        self.vocab = sorted(set(all_tokens))
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        print(f"Vocabulary size: {self.vocab_size}")
        print(f"Total tokens: {len(all_tokens)}")
        print(f"Sample tokens: {all_tokens[:20]}")
        
        # Create sequences with minimum length check
        min_seq_length = min(seq_length, len(all_tokens) - 1)
        if min_seq_length < 50:  # Need at least some reasonable sequence length
            raise ValueError(f"Not enough tokens ({len(all_tokens)}) to create meaningful sequences")
        
        # Create sequences
        step_size = max(1, seq_length // 4)  # Ensure step_size is at least 1
        for i in range(0, len(all_tokens) - min_seq_length, step_size):
            seq = all_tokens[i:i + min_seq_length + 1]  # +1 for target
            if len(seq) == min_seq_length + 1:
                # Convert to vocab indices
                indexed_seq = [self.token_to_id[token] for token in seq]
                self.sequences.append(indexed_seq)
        
        print(f"Created {len(self.sequences)} sequences")
        
        if len(self.sequences) == 0:
            raise ValueError("No sequences created! Check sequence length and token data.")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:x.size(0), :]

class MusicTransformer(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_layers=6, dim_feedforward=2048, max_len=512):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.pos_encoding(src.transpose(0, 1)).transpose(0, 1)
        
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        output = self.transformer(src, src_mask)
        output = self.output_projection(output)
        return output

def train_model(model, dataloader, num_epochs=10, lr=1e-4):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, num_epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Assuming 0 is PAD token
    
    for epoch in range(num_epochs):
        total_loss = 0
        progress_bar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src)
            
            loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            
            optimizer.step()
            total_loss += loss.item()
            
            if batch_idx % 100 == 0:
                progress_bar.set_postfix({'loss': loss.item()})
        
        scheduler.step()
        avg_loss = total_loss / len(dataloader)
        print(f'Epoch {epoch+1}, Average Loss: {avg_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
            }, f'music_transformer_epoch_{epoch+1}.pt')

def generate_music(model, dataset, seed_length=50, generate_length=500, temperature=0.8, top_k=50):
    model.eval()
    
    # Start with a random seed from the dataset
    seed_idx = random.randint(0, len(dataset) - 1)
    seed_sequence, _ = dataset[seed_idx]
    seed_sequence = seed_sequence[:seed_length].unsqueeze(0).to(device)
    
    generated = seed_sequence.clone()
    
    with torch.no_grad():
        for _ in tqdm(range(generate_length), desc="Generating music"):
            # Get the last sequence_length tokens
            current_seq = generated[:, -model.pos_encoding.pe.size(0):]
            
            # Forward pass
            output = model(current_seq)
            logits = output[0, -1, :] / temperature
            
            # Apply top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits[top_k_indices] = top_k_logits
            
            # Sample next token
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    return generated[0].cpu().numpy()

def tokens_to_midi(tokens, dataset, output_path="generated_music.mid"):
    """Convert tokens back to MIDI using music21"""
    # Convert token IDs back to original tokens
    original_tokens = [dataset.id_to_token[token_id] for token_id in tokens if token_id in dataset.id_to_token]
    
    # Create a music21 stream
    music_stream = stream.Stream()
    music_stream.append(tempo.TempoIndication(number=120))
    music_stream.append(meter.TimeSignature('4/4'))
    
    i = 0
    current_chord_notes = []
    
    while i < len(original_tokens):
        token = original_tokens[i]
        
        if token == 1:  # BOS
            i += 1
            continue
        elif token == 2:  # EOS
            break
        elif 1000 <= token < 2000:  # Note pitch
            pitch_midi = token - 1000
            # Look ahead for duration
            if i + 1 < len(original_tokens) and 2000 <= original_tokens[i + 1] < 3000:
                duration_token = original_tokens[i + 1]
                note_duration = (duration_token - 2000) / 4.0
                
                if current_chord_notes:
                    # This is part of a chord
                    current_chord_notes.append(pitch_midi)
                else:
                    # Single note
                    music_note = note.Note(pitch_midi)
                    music_note.duration = duration.Duration(quarterLength=max(0.25, note_duration))
                    music_stream.append(music_note)
                i += 2
            else:
                i += 1
        elif 3000 <= token < 4000:  # Chord duration
            if current_chord_notes:
                chord_duration = (token - 3000) / 4.0
                music_chord = chord.Chord(current_chord_notes)
                music_chord.duration = duration.Duration(quarterLength=max(0.25, chord_duration))
                music_stream.append(music_chord)
                current_chord_notes = []
            i += 1
        else:
            i += 1
    
    # Write to MIDI file
    music_stream.write('midi', fp=output_path)
    print(f"Generated MIDI saved to {output_path}")

def main():
    # Load dataset
    print("Loading dataset...")
    dataset = MusicDataset('/home/mog014/student_files/music/data/tokenized', seq_length=256)
    
    # Save dataset vocab for later use
    with open('music_vocab.pkl', 'wb') as f:
        pickle.dump({
            'vocab': dataset.vocab,
            'token_to_id': dataset.token_to_id,
            'id_to_token': dataset.id_to_token,
            'vocab_size': dataset.vocab_size
        }, f)
    
    # Create dataloader - reduce num_workers if having issues
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True, num_workers=0)  # Set num_workers=0 for debugging
    
    # Initialize model
    model = MusicTransformer(
        vocab_size=dataset.vocab_size,
        d_model=512,
        nhead=8,
        num_layers=6,
        dim_feedforward=2048,
        max_len=512
    ).to(device)
    
    print(f"Model has {sum(p.numel() for p in model.parameters())} parameters")
    
    # Train model
    print("Starting training...")
    train_model(model, dataloader, num_epochs=20, lr=1e-4)
    
    # Generate music
    print("Generating music...")
    generated_tokens = generate_music(model, dataset, seed_length=50, generate_length=1000)
    
    # Convert to MIDI
    tokens_to_midi(generated_tokens, dataset, "generated_music.mid")
    
    print("Training and generation complete!")

if __name__ == "__main__":
    main()
