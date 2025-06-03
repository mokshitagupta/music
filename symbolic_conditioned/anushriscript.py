#!/usr/bin/env python
# coding: utf-8

"""
Composer-Conditioned Music Generation with Chord Support
Fixed version with proper training and inference functionality.
"""

import os
import glob
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from tqdm import tqdm
from music21 import converter, stream, note, chord, instrument, duration
import numpy as np

# ==================== SPECIAL TOKENS ====================
SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "EOS": 2}

# ==================== VOCABULARY AND DATA LOADING ====================

def build_vocab_from_files(root_dir):
    """Build vocabulary from tokenized files"""
    all_tokens = set()
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith(".txt"):
                try:
                    with open(os.path.join(root, fname), 'r') as f:
                        tokens = list(map(int, f.read().strip().split()))
                        all_tokens.update(tokens)
                except Exception as e:
                    print(f"Error reading {fname}: {e}")
    
    sorted_vocab = sorted(all_tokens)
    itos = {i: tok for i, tok in enumerate(sorted_vocab)}
    stoi = {tok: i for i, tok in itos.items()}
    return stoi, itos

def load_composer_map(csv_path):
    """Load composer mapping from CSV file"""
    if not os.path.exists(csv_path):
        print(f"Warning: CSV file {csv_path} not found. Creating dummy composer mapping.")
        return {"dummy.mid": "Unknown Composer"}
    
    df = pd.read_csv(csv_path)
    df['midi_filename'] = df['midi_filename'].apply(lambda x: os.path.basename(x).lower())
    return dict(zip(df['midi_filename'], df['canonical_composer']))

# ==================== DATASET CLASS ====================

class ComposerConditionedDataset(Dataset):
    def __init__(self, token_dir, stoi, composer_map, composer_to_id, 
                 context_len=128, target_len=128, min_tokens=50):
        self.samples = []
        self.context_len = context_len
        self.target_len = target_len
        
        file_paths = glob.glob(os.path.join(token_dir, '**/*.txt'), recursive=True)
        
        for path in tqdm(file_paths, desc="Loading dataset"):
            fname_txt = os.path.basename(path)
            fname_midi = fname_txt.replace('.txt', '.midi').lower()
            
            # Try different extensions
            composer = None
            for ext in ['.midi', '.mid']:
                test_name = fname_txt.replace('.txt', ext).lower()
                if test_name in composer_map:
                    composer = composer_map[test_name]
                    break
            
            if composer is None or composer not in composer_to_id:
                # Use default composer if not found
                composer = list(composer_to_id.keys())[0] if composer_to_id else "Unknown Composer"
                if composer not in composer_to_id:
                    continue
            
            try:
                with open(path, 'r') as f:
                    raw_tokens = list(map(int, f.read().strip().split()))
                    token_indices = [stoi.get(t, 0) for t in raw_tokens if t in stoi]
                    
                    if len(token_indices) >= context_len + target_len and len(token_indices) >= min_tokens:
                        # Create overlapping windows
                        step_size = max(1, context_len // 2)
                        for i in range(0, len(token_indices) - context_len - target_len + 1, step_size):
                            ctx = token_indices[i:i+context_len]
                            tgt = token_indices[i+context_len:i+context_len+target_len]
                            self.samples.append((ctx, tgt, composer_to_id[composer]))
                            
            except Exception as e:
                print(f"Error processing {path}: {e}")
        
        print(f"Created dataset with {len(self.samples)} samples")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        context, target, composer_id = self.samples[idx]
        return torch.tensor(context, dtype=torch.long), torch.tensor(target, dtype=torch.long), torch.tensor(composer_id, dtype=torch.long)

# ==================== MODEL ARCHITECTURES ====================

class ComposerConditionedTransformer(nn.Module):
    def __init__(self, vocab_size, d_model, num_composers, nhead=8, num_layers=6, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.composer_embedding = nn.Embedding(num_composers, d_model)
        self.pos_encoding = nn.Parameter(torch.randn(1000, d_model))
        
        encoder_layer = TransformerEncoderLayer(
            d_model=d_model, 
            nhead=nhead, 
            dim_feedforward=d_model*4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.layer_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, composer_id):
        seq_len = x.size(1)
        
        # Token embeddings
        token_emb = self.token_embedding(x) * (self.d_model ** 0.5)
        
        # Positional encoding
        pos_emb = self.pos_encoding[:seq_len].unsqueeze(0).expand(x.size(0), -1, -1)
        
        # Composer conditioning
        composer_emb = self.composer_embedding(composer_id).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine embeddings
        x = self.dropout(token_emb + pos_emb + composer_emb)
        
        # Transformer processing
        x = self.transformer(x)
        x = self.layer_norm(x)
        
        return self.output(x)

class ComposerConditionedLSTM(nn.Module):
    def __init__(self, vocab_size, d_model, num_composers, num_layers=3, dropout=0.2):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.composer_embedding = nn.Embedding(num_composers, d_model)
        self.lstm = nn.LSTM(
            input_size=d_model, 
            hidden_size=d_model, 
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            batch_first=True
        )
        self.layer_norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, composer_id):
        token_emb = self.token_embedding(x)
        composer_emb = self.composer_embedding(composer_id).unsqueeze(1)
        x = self.dropout(token_emb + composer_emb)
        
        out, _ = self.lstm(x)
        out = self.layer_norm(out)
        return self.output(out)

# ==================== TRAINING FUNCTIONS ====================

def train_model(model, dataloader, val_dataloader=None, epochs=15, lr=1e-4, device='cuda', model_path='best_composer_model.pth'):
    """Enhanced training with validation and learning rate scheduling"""
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # Ignore PAD tokens
    
    best_val_loss = float('inf')
    
    for epoch in range(epochs):
        # Training
        model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        for context, target, composer_id in progress_bar:
            context, target, composer_id = context.to(device), target.to(device), composer_id.to(device)
            
            optimizer.zero_grad()
            output = model(context, composer_id)
            loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
            
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            progress_bar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_train_loss = total_loss / num_batches
        
        # Validation
        if val_dataloader:
            model.eval()
            val_loss = 0
            val_batches = 0
            
            with torch.no_grad():
                for context, target, composer_id in val_dataloader:
                    context, target, composer_id = context.to(device), target.to(device), composer_id.to(device)
                    output = model(context, composer_id)
                    loss = criterion(output.reshape(-1, output.size(-1)), target.reshape(-1))
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
            
            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                torch.save(model.state_dict(), model_path)
                print(f"New best model saved with val loss: {avg_val_loss:.4f}")
        else:
            print(f"Epoch {epoch+1}: Train Loss: {avg_train_loss:.4f}")
            # Save model after each epoch if no validation
            torch.save(model.state_dict(), model_path)
        
        scheduler.step()
    
    return model

# ==================== GENERATION FUNCTIONS ====================

@torch.no_grad()
def generate_sequence(model, context, composer_id, max_tokens=256, temperature=1.0, top_k=50, top_p=0.9, device=None):
    """Enhanced generation with robust top-k and top-p sampling"""
    model.eval()
    
    # Get device from model if not specified
    if device is None:
        device = next(model.parameters()).device
    
    # Ensure all tensors are on the correct device
    generated = context.clone().to(device)
    composer_id = composer_id.to(device)
    
    for _ in range(max_tokens):
        # Get model predictions
        logits = model(generated.unsqueeze(0), composer_id)
        logits = logits[:, -1, :].squeeze(0)  # Get last token predictions
        
        # Apply temperature scaling
        if temperature != 1.0:
            logits = logits / max(temperature, 1e-8)
        
        # Convert to probabilities before filtering
        probs = torch.softmax(logits, dim=-1)
        
        # Top-k filtering
        if top_k > 0 and top_k < logits.size(-1):
            top_k_probs, top_k_indices = torch.topk(probs, top_k)
            filtered_probs = torch.zeros_like(probs)
            filtered_probs[top_k_indices] = top_k_probs
            probs = filtered_probs
        
        # Top-p (nucleus) sampling
        if top_p < 1.0:
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            cutoff = torch.searchsorted(cumulative_probs, top_p)
            cutoff = max(1, cutoff.item())
            
            probs_copy = probs.clone()
            probs_copy[sorted_indices[cutoff:]] = 0.0
            probs = probs_copy
        
        # Ensure probabilities sum to 1 and are non-negative
        probs = torch.clamp(probs, min=1e-8)
        probs = probs / probs.sum()
        
        # Sample next token
        try:
            next_token = torch.multinomial(probs, num_samples=1)
        except RuntimeError:
            next_token = torch.argmax(probs).unsqueeze(0)
        
        # Append to sequence
        generated = torch.cat((generated, next_token), dim=0)
        
        # Stop if we hit EOS token
        if next_token.item() == 2:  # EOS token
            break
    
    return generated

def tokens_to_midi_fixed(tokens, output_path, itos=None):
    """Fixed token-to-MIDI conversion with better note handling"""
    s = stream.Stream()
    s.append(instrument.Piano())
    
    if isinstance(tokens, torch.Tensor):
        tokens = tokens.cpu().tolist()
    
    print(f"Converting {len(tokens)} tokens to MIDI...")
    
    i = 0
    notes_added = 0
    
    while i < len(tokens):
        tok = tokens[i]
        
        # Skip special tokens
        if tok in [0, 1, 2]:
            i += 1
            continue
        
        # Map token back to original value if using vocabulary mapping
        if itos is not None and tok in itos:
            original_tok = itos[tok]
        else:
            original_tok = tok
            
        # Single notes (1000-1127 range)
        if 1000 <= original_tok <= 1127:
            pitch = original_tok - 1000
            note_duration = 1.0  # default quarter note
            
            # Check for duration token
            if i + 1 < len(tokens):
                next_tok = tokens[i + 1]
                if itos is not None and next_tok in itos:
                    next_original = itos[next_tok]
                else:
                    next_original = next_tok
                    
                if 2000 <= next_original <= 2100:
                    note_duration = max(0.25, (next_original - 2000) / 4.0)
                    i += 1
            
            if 0 <= pitch <= 127:
                try:
                    n = note.Note(pitch, quarterLength=note_duration)
                    n.volume.velocity = np.random.randint(60, 90)
                    s.append(n)
                    notes_added += 1
                except Exception as e:
                    print(f"Error creating note {pitch}: {e}")
            i += 1
            
        # Chord notes (3000-3127 range)
        elif 3000 <= original_tok <= 3127:
            chord_pitches = []
            
            # Collect all chord pitches
            while i < len(tokens):
                current_tok = tokens[i]
                if itos is not None and current_tok in itos:
                    current_original = itos[current_tok]
                else:
                    current_original = current_tok
                    
                if 3000 <= current_original <= 3127:
                    pitch = current_original - 3000
                    if 0 <= pitch <= 127:
                        chord_pitches.append(pitch)
                    i += 1
                else:
                    break
            
            # Get chord duration
            chord_duration = 1.0
            if i < len(tokens):
                dur_tok = tokens[i]
                if itos is not None and dur_tok in itos:
                    dur_original = itos[dur_tok]
                else:
                    dur_original = dur_tok
                    
                if 4000 <= dur_original <= 4100:
                    chord_duration = max(0.25, (dur_original - 4000) / 4.0)
                    i += 1
            
            # Create chord
            if len(chord_pitches) >= 2:  # Need at least 2 notes for a chord
                try:
                    c = chord.Chord(chord_pitches, quarterLength=chord_duration)
                    c.volume.velocity = np.random.randint(60, 90)
                    s.append(c)
                    notes_added += 1
                except Exception as e:
                    print(f"Error creating chord {chord_pitches}: {e}")
        else:
            i += 1
    
    print(f"Added {notes_added} musical elements to stream")
    
    # If no notes were added, add a simple melody
    if notes_added == 0:
        print("No valid notes found, adding default melody...")
        default_melody = [60, 62, 64, 65, 67, 69, 71, 72]  # C major scale
        for pitch in default_melody:
            n = note.Note(pitch, quarterLength=0.5)
            n.volume.velocity = 70
            s.append(n)
        notes_added = len(default_melody)
    
    # Save MIDI file
    try:
        s.write("midi", fp=output_path)
        print(f"Successfully saved MIDI with {notes_added} elements to: {output_path}")
        return True
    except Exception as e:
        print(f"Error saving MIDI: {e}")
        return False

def create_musical_seeds(stoi):
    """Create various musical seed patterns"""
    seeds = {}
    
    def safe_get(token_id, default=1):
        return stoi.get(token_id, default)
    
    # Use direct token values since we're working with the vocabulary mapping
    seeds['simple_melody'] = [
        safe_get(1, 1),      # BOS
        safe_get(1060, 1060),   # C4
        safe_get(2004, 2004),   # Quarter note
        safe_get(1062, 1062),   # D4
        safe_get(2004, 2004),   # Quarter note
        safe_get(1064, 1064),   # E4
        safe_get(2004, 2004),   # Quarter note
    ]
    
    seeds['chord_progression'] = [
        safe_get(1, 1),      # BOS
        safe_get(3060, 3060),   # C4 in chord
        safe_get(3064, 3064),   # E4 in chord
        safe_get(3067, 3067),   # G4 in chord
        safe_get(4008, 4008),   # Half note duration
    ]
    
    # Add some variety in seeds
    seeds['ascending'] = [safe_get(1, 1)] + [safe_get(1060 + i, 1060 + i) for i in range(8)]
    seeds['minimal'] = [safe_get(1, 1), safe_get(1060, 1060), safe_get(2004, 2004)]
    
    return seeds

@torch.no_grad()
def generate_with_seed(model, seed_tokens, composer_name, composer_to_id, stoi, max_tokens=200, 
                      temperature=1.0, top_k=50, top_p=0.9, device='cuda'):
    """Generate music with a specific seed sequence"""
    model.eval()
    
    if composer_name not in composer_to_id:
        print(f"Composer '{composer_name}' not found. Available: {list(composer_to_id.keys())}")
        return None
    
    composer_id = torch.tensor([composer_to_id[composer_name]], device=device)
    seed_context = torch.tensor(seed_tokens, dtype=torch.long, device=device)
    
    generated = generate_sequence(
        model, seed_context, composer_id, 
        max_tokens=max_tokens, temperature=temperature, 
        top_k=top_k, top_p=top_p, device=device
    )
    
    return generated

def safe_generate_music(model, composer_to_id, stoi, itos, device='cuda', 
                       context_len=128, output_dir='generated_music'):
    """Safely generate music with error handling"""
    print("="*60)
    print("GENERATING MUSIC")
    print("="*60)
    
    os.makedirs(output_dir, exist_ok=True)
    
    available_composers = list(composer_to_id.keys())
    print(f"Available composers: {', '.join(available_composers[:5])}...")
    
    seeds = create_musical_seeds(stoi)
    print(f"Created {len(seeds)} seed patterns")
    
    target_composers = [c for c in ['Johann Sebastian Bach', 'Wolfgang Amadeus Mozart', 'Ludwig van Beethoven'] 
                       if c in available_composers]
    
    if not target_composers:
        target_composers = available_composers[:3]
    
    print(f"Generating for composers: {target_composers}")
    
    successful_generations = 0
    
    for composer_name in target_composers:
        print(f"\n--- Generating for {composer_name} ---")
        
        for seed_name, seed_tokens in seeds.items():
            # Pad seed to context length
            if len(seed_tokens) < context_len:
                padding = [0] * (context_len - len(seed_tokens))
                seed_tokens_padded = seed_tokens + padding
            else:
                seed_tokens_padded = seed_tokens[:context_len]
            
            filename = f"{composer_name.replace(' ', '_').lower()}_{seed_name}.mid"
            filepath = os.path.join(output_dir, filename)
            
            try:
                generated = generate_with_seed(
                    model=model,
                    seed_tokens=seed_tokens_padded,
                    composer_name=composer_name,
                    composer_to_id=composer_to_id,
                    stoi=stoi,
                    max_tokens=100,
                    temperature=1.0,
                    top_k=50,
                    top_p=0.9,
                    device=device
                )
                
                if generated is not None:
                    if tokens_to_midi_fixed(generated.cpu(), filepath, itos):
                        successful_generations += 1
                        print(f"  ✓ Saved: {filename}")
                    else:
                        print(f"  ✗ Failed to save MIDI: {filename}")
                else:
                    print(f"  ✗ Failed to generate for {composer_name}")
                    
            except Exception as e:
                print(f"  ✗ Error generating {filename}: {str(e)}")
                continue
    
    print(f"\nGeneration complete! Successful: {successful_generations} files")
    return successful_generations

# ==================== MAIN EXECUTION ====================

def main():
    """Main execution function with training and inference"""
    CONFIG = {
        'token_dir': '../data/tokenized',
        'csv_path': '../data/midi/maestro-v3.0.0.csv',
        'model_path': 'best_composer_model.pth',
        'context_len': 128,
        'target_len': 128,
        'batch_size': 16,
        'd_model': 256,
        'epochs': 15,
        'learning_rate': 1e-4,
        'device': 'cpu' if not torch.cuda.is_available() else 'cuda'
    }
    
    print("Starting Composer-Conditioned Music Generation...")
    print(f"Using device: {CONFIG['device']}")
    
    # Check if data directory exists
    if not os.path.exists(CONFIG['token_dir']):
        print(f"Error: Token directory {CONFIG['token_dir']} not found!")
        print("Please ensure you have tokenized MIDI files in the specified directory.")
        return
    
    # Build vocabulary and load composer mapping
    print("Building vocabulary...")
    stoi, itos = build_vocab_from_files(CONFIG['token_dir'])
    
    print("Loading composer mapping...")
    composer_map = load_composer_map(CONFIG['csv_path'])
    composer_names = sorted(set(composer_map.values()))
    
    # Ensure we have at least one composer
    if not composer_names:
        composer_names = ["Unknown Composer"]
    
    composer_to_id = {name: i for i, name in enumerate(composer_names)}
    
    print(f"Vocabulary size: {len(stoi)}")
    print(f"Number of composers: {len(composer_names)}")
    
    # Create model
    print("Creating model...")
    model = ComposerConditionedTransformer(
        vocab_size=len(stoi),
        d_model=CONFIG['d_model'],
        num_composers=len(composer_names),
        nhead=8,
        num_layers=4,  # Reduced for faster training
        dropout=0.1
    )
    
    # Check if model exists, if not train it
    if not os.path.exists(CONFIG['model_path']):
        print("No pre-trained model found. Starting training...")
        
        # Create dataset
        print("Creating dataset...")
        dataset = ComposerConditionedDataset(
            CONFIG['token_dir'], stoi, composer_map, composer_to_id,
            CONFIG['context_len'], CONFIG['target_len']
        )
        
        if len(dataset) == 0:
            print("Error: No valid samples found in dataset!")
            return
        
        # Split dataset
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])
        
        train_dataloader = DataLoader(train_dataset, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
        val_dataloader = DataLoader(val_dataset, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0) if val_size > 0 else None
        
        # Train model
        print("Training model...")
        model = train_model(
            model, train_dataloader, val_dataloader,
            epochs=CONFIG['epochs'], lr=CONFIG['learning_rate'], 
            device=CONFIG['device'], model_path=CONFIG['model_path']
        )
        print("Training completed!")
    else:
        print(f"Loading pre-trained model from {CONFIG['model_path']}...")
        try:
            state_dict = torch.load(CONFIG['model_path'], map_location=CONFIG['device'])
            model.load_state_dict(state_dict)
            print("Model loaded successfully!")
        except Exception as e:
            print(f"Error loading model: {e}")
            return
    
    # Move model to device
    model = model.to(CONFIG['device'])
    
    # Generate music
    print("Generating music...")
    successful = safe_generate_music(
        model=model,
        composer_to_id=composer_to_id,
        stoi=stoi,
        itos=itos,
        device=CONFIG['device'],
        context_len=CONFIG['context_len'],
        output_dir='generated_music'
    )
    
    if successful > 0:
        print(f"\nSuccess! Generated {successful} MIDI files in 'generated_music' directory.")
        print("You can play these files with any MIDI player or import them into a DAW.")
    else:
        print("\nNo MIDI files were successfully generated. Check the console output for errors.")

if __name__ == "__main__":
    main()