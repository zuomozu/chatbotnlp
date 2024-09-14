import pandas as pd
import json
import re
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
from rouge_score import rouge_scorer
from bert_score import score

# Evaluate the model
def evaluate_model(model, test_dataset, tokenizer):
    model.eval()  # Set the model to evaluation mode

    all_predictions = []
    all_labels = []

    # Process the dataset in batches
    for i in range(0, len(test_dataset), 16):  # Batch size of 16
        batch = test_dataset[i:i+16]

        input_ids = torch.tensor(batch['input_ids']).to(model.device)
        attention_mask = torch.tensor(batch['attention_mask']).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_length=128,
                num_return_sequences=1,
                decoder_start_token_id=model.config.decoder_start_token_id,
                bos_token_id=model.config.bos_token_id,
                eos_token_id=model.config.eos_token_id,
                pad_token_id=model.config.pad_token_id,
            )

        # Decode predictions
        decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_predictions.extend(decoded_preds)

        # Decode labels
        decoded_labels = tokenizer.batch_decode(batch['labels'], skip_special_tokens=True)
        all_labels.extend(decoded_labels)

    # ROUGE-L Score
    scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
    rouge_scores = [scorer.score(pred, label)['rougeL'].fmeasure for pred, label in zip(all_predictions, all_labels)]
    avg_rouge_l = sum(rouge_scores) / len(rouge_scores)

    # BERT Score
    P, R, F1 = score(all_predictions, all_labels, lang='en', verbose=True)
    avg_bert_score = F1.mean().item()

    return avg_rouge_l, avg_bert_score

# Evaluate the model
rouge_l, bert_score = evaluate_model(model, test_dataset, tokenizer)
print(f"ROUGE-L Score: {rouge_l}")
print(f"BERT Score: {bert_score}")
