from music21 import converter, note, chord, instrument
import os

SPECIAL_TOKENS = {"PAD": 0, "BOS": 1, "EOS": 2}

def encode(midi_path):
    midi = converter.parse(midi_path)
    tokens = [SPECIAL_TOKENS["BOS"]]
    for el in midi.flat.notes:
        if isinstance(el, note.Note):
            tokens.append(1000 + el.pitch.midi)
            tokens.append(2000 + int(el.quarterLength * 4))
        elif isinstance(el, chord.Chord):
            for n in el.pitches:
                tokens.append(1000 + n.midi)
            tokens.append(3000 + int(el.quarterLength * 4))
    tokens.append(SPECIAL_TOKENS["EOS"])
    return tokens

def decode(tokens, itos):
    events = []
    for tok in tokens:
        if tok in itos:
            events.append(itos[tok])
    return events

def build_vocab(token_lists):
    vocab = {tok for seq in token_lists for tok in seq}
    itos = {i: tok for i, tok in enumerate(sorted(vocab))}
    stoi = {tok: i for i, tok in itos.items()}
    return stoi, itos

def tokenize_all(midi_root, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for root, _, files in os.walk(midi_root):
        for fname in files:
            if fname.endswith(".mid") or fname.endswith(".midi"):
                midi_path = os.path.join(root, fname)
                try:
                    tokens = encode(midi_path)
                    rel_path = os.path.relpath(midi_path, midi_root)
                    out_path = os.path.join(output_dir, rel_path.replace(".midi", ".txt")).replace(".mid", ".txt")
                    os.makedirs(os.path.dirname(out_path), exist_ok=True)
                    with open(out_path, 'w') as f:
                        f.write(" ".join(map(str, tokens)))
                except Exception as e:
                    print(f"Failed to tokenize {midi_path}: {e}")



if __name__ == "__main__":
    tokenize_all("data/midi", "data/tokenized")