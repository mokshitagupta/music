import torch
from transformer import MusicTransformer
from midi_tokenizer import build_vocab, encode, decode
from events_to_midi import events_to_midi_polyphonic
import random
import yaml

with open("config.yaml") as f:
    cfg = yaml.safe_load(f)

model = MusicTransformer(cfg["vocab_size"])
model.load_state_dict(torch.load(cfg["save_path"]))
model.eval()

seed = [random.randint(0, cfg["vocab_size"]-1) for _ in range(16)]
input_seq = torch.tensor(seed).unsqueeze(0)

generated = seed[:]
for _ in range(cfg["generate_len"]):
    logits = model(input_seq)
    next_token = torch.argmax(logits[0, -1]).item()
    generated.append(next_token)
    input_seq = torch.tensor(generated[-cfg["seq_len"]:]).unsqueeze(0)

with open(cfg["itos"], 'r') as f:
    itos = eval(f.read())

events = decode(generated, itos)
events_to_midi_polyphonic(events, cfg["output_path"])
