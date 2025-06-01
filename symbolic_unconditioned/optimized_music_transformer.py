import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import os
import numpy as np
from tqdm import tqdm
import math
import random
from music21 import stream, note, chord, tempo, meter, duration, key
import pickle
from collections import Counter
import matplotlib.pyplot as plt

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

class MusicDataset(Dataset):
    def __init__(self, tokenized_dir, seq_length=256, train=True, val_split=0.15):
        self.seq_length = seq_length
        self.sequences = []
        
        if not os.path.exists(tokenized_dir):
            raise FileNotFoundError(f"Tokenized directory not found: {tokenized_dir}")
        
        # Load tokens
        all_tokens = []
        file_count = 0
        
        for root, dirs, files in os.walk(tokenized_dir):
            for file in files:
                if file.endswith('.txt'):
                    file_count += 1
                    path = os.path.join(root, file)
                    try:
                        with open(path, 'r') as f:
                            content = f.read().strip()
                            if content:
                                tokens = list(map(int, content.split()))
                                all_tokens.extend(tokens)
                    except Exception as e:
                        print(f"Error loading {path}: {e}")
        
        if len(all_tokens) == 0:
            raise ValueError("No tokens found!")
        
        # Build vocabulary
        self.vocab = sorted(set(all_tokens))
        self.vocab_size = len(self.vocab)
        self.token_to_id = {token: i for i, token in enumerate(self.vocab)}
        self.id_to_token = {i: token for token, i in self.token_to_id.items()}
        
        print(f"Vocab size: {self.vocab_size}, Total tokens: {len(all_tokens)}")
        
        # Create sequences with aggressive overlap for faster convergence
        step_size = seq_length // 4  # 75% overlap for more training data
        for i in range(0, len(all_tokens) - seq_length, step_size):
            seq = all_tokens[i:i + seq_length + 1]
            if len(seq) == seq_length + 1:
                indexed_seq = [self.token_to_id[token] for token in seq]
                self.sequences.append(indexed_seq)
        
        # Train/val split
        random.shuffle(self.sequences)
        split_idx = int(len(self.sequences) * (1 - val_split))
        
        if train:
            self.sequences = self.sequences[:split_idx]
            print(f"Training sequences: {len(self.sequences)}")
        else:
            self.sequences = self.sequences[split_idx:]
            print(f"Validation sequences: {len(self.sequences)}")
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq = self.sequences[idx]
        return torch.tensor(seq[:-1], dtype=torch.long), torch.tensor(seq[1:], dtype=torch.long)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=1000):
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
    def __init__(self, vocab_size, d_model=384, nhead=6, num_layers=4, dim_feedforward=1536, max_len=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoding = PositionalEncoding(d_model, max_len)
        self.dropout = nn.Dropout(dropout)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
            activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Improved weight initialization
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
    
    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz)) == 1
        mask = mask.transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask
    
    def forward(self, src, src_mask=None):
        src = self.embedding(src) * math.sqrt(self.d_model)
        src = self.dropout(self.pos_encoding(src.transpose(0, 1)).transpose(0, 1))
        
        if src_mask is None:
            src_mask = self.generate_square_subsequent_mask(src.size(1)).to(src.device)
        
        output = self.transformer(src, src_mask)
        output = self.layer_norm(output)
        output = self.output_projection(output)
        return output

def save_checkpoint(model, optimizer, scheduler, epoch, train_loss, val_loss, checkpoint_dir='checkpoints'):
    os.makedirs(checkpoint_dir, exist_ok=True)
    checkpoint_path = os.path.join(checkpoint_dir, f'music_transformer_epoch_{epoch+1}.pt')
    
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, checkpoint_path)
    
    # Save as latest and best
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'train_loss': train_loss,
        'val_loss': val_loss,
    }, latest_path)

def load_checkpoint(model, optimizer, scheduler, checkpoint_path):
    if not os.path.exists(checkpoint_path):
        return 0, float('inf'), float('inf')
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    if 'scheduler_state_dict' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    
    epoch = checkpoint['epoch']
    train_loss = checkpoint.get('train_loss', float('inf'))
    val_loss = checkpoint.get('val_loss', float('inf'))
    
    print(f"Resumed from epoch {epoch+1} - Train: {train_loss:.4f}, Val: {val_loss:.4f}")
    return epoch + 1, train_loss, val_loss

def find_latest_checkpoint(checkpoint_dir='checkpoints'):
    if not os.path.exists(checkpoint_dir):
        return None
    
    latest_path = os.path.join(checkpoint_dir, 'latest_checkpoint.pt')
    if os.path.exists(latest_path):
        return latest_path
    return None

def compute_perplexity(model, dataloader, max_batches=30):
    """Quick perplexity calculation"""
    model.eval()
    total_loss = 0
    total_tokens = 0
    criterion = nn.CrossEntropyLoss(reduction='sum')
    
    with torch.no_grad():
        for i, (src, tgt) in enumerate(dataloader):
            if i >= max_batches:
                break
            src, tgt = src.to(device), tgt.to(device)
            output = model(src)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            total_loss += loss.item()
            total_tokens += tgt.numel()
    
    return math.exp(total_loss / total_tokens) if total_tokens > 0 else float('inf')

def train_model(model, train_loader, val_loader, num_epochs=15, lr=5e-4, warmup_steps=1000):
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01, betas=(0.9, 0.95))
    
    # Warmup + cosine annealing scheduler
    total_steps = len(train_loader) * num_epochs
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr=lr,
        total_steps=total_steps,
        pct_start=0.1,  # 10% warmup
        div_factor=10,
        final_div_factor=100
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=0, label_smoothing=0.1)
    
    best_val_loss = float('inf')
    train_losses, val_losses = [], []
    patience_counter = 0
    
    # Try to resume
    checkpoint_path = find_latest_checkpoint()
    start_epoch = 0
    if checkpoint_path:
        start_epoch, _, best_val_loss = load_checkpoint(model, optimizer, scheduler, checkpoint_path)
    
    print(f"Starting training from epoch {start_epoch+1}")
    
    for epoch in range(start_epoch, num_epochs):
        # Training
        model.train()
        total_train_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}')
        
        for batch_idx, (src, tgt) in enumerate(progress_bar):
            src, tgt = src.to(device), tgt.to(device)
            
            optimizer.zero_grad()
            output = model(src)
            loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            optimizer.step()
            scheduler.step()
            
            total_train_loss += loss.item()
            
            if batch_idx % 100 == 0:
                progress_bar.set_postfix({
                    'loss': f'{loss.item():.3f}', 
                    'lr': f'{scheduler.get_last_lr()[0]:.2e}'
                })
        
        # Validation
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for src, tgt in val_loader:
                src, tgt = src.to(device), tgt.to(device)
                output = model(src)
                loss = criterion(output.reshape(-1, output.size(-1)), tgt.reshape(-1))
                total_val_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)
        
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        print(f'Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}')
        
        # Save checkpoint every 2 epochs or if best
        if (epoch + 1) % 2 == 0 or avg_val_loss < best_val_loss:
            save_checkpoint(model, optimizer, scheduler, epoch, avg_train_loss, avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                print(f"New best model! Val loss: {avg_val_loss:.4f}")
            else:
                patience_counter += 1
        
        # Early stopping
        if patience_counter >= 4:
            print(f"Early stopping at epoch {epoch+1}")
            break
        
        # Compute perplexity every 3 epochs
        if (epoch + 1) % 3 == 0:
            train_ppl = compute_perplexity(model, train_loader)
            val_ppl = compute_perplexity(model, val_loader)
            print(f"Perplexity - Train: {train_ppl:.2f}, Val: {val_ppl:.2f}")
    
    # Plot training curves
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', alpha=0.8)
    plt.plot(val_losses, label='Val Loss', alpha=0.8)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training Progress')
    plt.grid(True, alpha=0.3)
    plt.savefig('training_progress.png', dpi=150, bbox_inches='tight')
    plt.close()
    
    return train_losses, val_losses

def apply_musical_constraints(logits, last_tokens, vocab, temperature=0.8):
    """Apply musical theory constraints"""
    if len(last_tokens) > 0:
        last_token = last_tokens[-1]
        if 1000 <= last_token < 2000:  # Last was a note
            current_pitch = last_token - 1000
            
            # Penalize large pitch jumps (>12 semitones)
            for i, token in enumerate(vocab):
                if 1000 <= token < 2000:
                    pitch_diff = abs(token - 1000 - current_pitch)
                    if pitch_diff > 12:
                        logits[i] *= 0.2
                    elif pitch_diff > 7:
                        logits[i] *= 0.5
    
    # Boost common durations
    duration_tokens = [i for i, token in enumerate(vocab) if 2000 <= token < 3000]
    if duration_tokens:
        for i in duration_tokens:
            token = vocab[i]
            duration_val = token - 2000
            if duration_val in [4, 8, 12, 16]:  # Quarter, half notes etc
                logits[i] *= 1.3
    
    return logits

def generate_music(model, dataset, seed_length=64, generate_length=600, temperature=0.85, top_k=40, checkpoint_path=None):
    if checkpoint_path and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint for generation")
    
    model.eval()
    
    # Find good seed starting with note
    for _ in range(50):
        seed_idx = random.randint(0, len(dataset) - 1)
        seed_sequence, _ = dataset[seed_idx]
        if seed_sequence[0] in range(1000, 2000):
            break
    
    seed_sequence = seed_sequence[:seed_length].unsqueeze(0).to(device)
    generated = seed_sequence.clone()
    
    with torch.no_grad():
        for step in tqdm(range(generate_length), desc="Generating"):
            # Use sliding window to stay within context
            max_context = min(256, model.pos_encoding.pe.size(0))
            current_seq = generated[:, -max_context:]
            
            output = model(current_seq)
            logits = output[0, -1, :] / temperature
            
            # Apply constraints
            last_tokens = generated[0, -5:].cpu().numpy().tolist()
            logits = apply_musical_constraints(logits, last_tokens, dataset.vocab, temperature)
            
            # Top-k sampling
            if top_k > 0:
                top_k_logits, top_k_indices = torch.topk(logits, top_k)
                logits = torch.full_like(logits, float('-inf'))
                logits[top_k_indices] = top_k_logits
            
            probs = torch.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, 1)
            generated = torch.cat([generated, next_token.unsqueeze(0)], dim=1)
    
    return generated[0].cpu().numpy()

def tokens_to_midi(tokens, dataset, output_path="generated_music.mid"):
    """Convert tokens to MIDI with better structure"""
    original_tokens = [dataset.id_to_token[token_id] for token_id in tokens if token_id in dataset.id_to_token]
    
    music_stream = stream.Stream()
    music_stream.append(tempo.TempoIndication(number=120))
    music_stream.append(meter.TimeSignature('4/4'))
    music_stream.append(key.KeySignature(0))  # C major
    
    i = 0
    current_time = 0
    
    while i < len(original_tokens):
        token = original_tokens[i]
        
        if token == 1:  # BOS
            i += 1
            continue
        elif token == 2:  # EOS
            break
        elif 1000 <= token < 2000:  # Note
            pitch_midi = max(36, min(96, token - 1000))  # Reasonable range
            
            # Look for duration
            duration_val = 1.0
            if i + 1 < len(original_tokens) and 2000 <= original_tokens[i + 1] < 3000:
                duration_token = original_tokens[i + 1]
                duration_val = max(0.25, min(4.0, (duration_token - 2000) / 4.0))
                i += 1
            
            music_note = note.Note(pitch_midi)
            music_note.duration = duration.Duration(quarterLength=duration_val)
            music_note.offset = current_time
            music_stream.append(music_note)
            current_time += duration_val
            
        i += 1
    
    try:
        music_stream.write('midi', fp=output_path)
        print(f"Generated MIDI saved to {output_path}")
    except Exception as e:
        print(f"Error saving MIDI: {e}")

def main():
    print("Loading datasets...")
    train_dataset = MusicDataset('data/tokenized', seq_length=256, train=True, val_split=0.15)
    val_dataset = MusicDataset('data/tokenized', seq_length=256, train=False, val_split=0.15)
    
    # Save vocab
    vocab_path = 'music_vocab.pkl'
    with open(vocab_path, 'wb') as f:
        pickle.dump({
            'vocab': train_dataset.vocab,
            'token_to_id': train_dataset.token_to_id,
            'id_to_token': train_dataset.id_to_token,
            'vocab_size': train_dataset.vocab_size
        }, f)
    
    # Optimized batch sizes and workers
    batch_size = 32 if torch.cuda.is_available() else 16
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                             num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, 
                           num_workers=2, pin_memory=True)
    
    # Balanced model - faster than v1, smarter than v2
    model = MusicTransformer(
        vocab_size=train_dataset.vocab_size,
        d_model=384,
        nhead=6,
        num_layers=4,
        dim_feedforward=1536,
        max_len=512,
        dropout=0.1
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")
    
    # Training
    if input("Train model? (y/n): ").lower() == 'y':
        print("Starting optimized training...")
        train_losses, val_losses = train_model(
            model, train_loader, val_loader, 
            num_epochs=15, lr=5e-4
        )
        print("Training complete!")
    
    # Generation
    print("Generating music...")
    generated_tokens = generate_music(
        model, train_dataset, 
        seed_length=64, 
        generate_length=800,
        temperature=0.85,
        top_k=40,
        checkpoint_path=find_latest_checkpoint()
    )
    
    tokens_to_midi(generated_tokens, train_dataset, "fast_generated_music.mid")
    print("Generation complete!")

if __name__ == "__main__":
    main()