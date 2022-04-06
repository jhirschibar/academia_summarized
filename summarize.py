from transformers import PegasusForConditionalGeneration, PegasusTokenizer
import torch
import pandas as pd

data = pd.read_json('training_data_v3.json', orient='index')

model_name = 'google/pegasus-arxiv'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
tokenizer = PegasusTokenizer.from_pretrained(model_name)
model = PegasusForConditionalGeneration.from_pretrained(model_name).to(device)
batch = tokenizer(src_text, truncation=True, padding='longest', return_tensors="pt").to(device)
summarized = model.generate(**batch)
tgt_text = tokenizer.batch_decode(summarized, skip_special_tokens=True)

print(tgt_text)