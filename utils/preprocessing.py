import pandas as pd
import json
import re
from transformers import BertTokenizer, EncoderDecoderModel, Trainer, TrainingArguments
from datasets import Dataset
from sklearn.model_selection import train_test_split
import torch
from rouge_score import rouge_scorer
from bert_score import score

# Function to fix common JSON formatting issues
def fix_json_string(json_str):
    # Ensure the JSON string is correctly formatted
    json_str = re.sub(r'\\', '', json_str)
    json_str = json_str.replace('"', "'")
    json_str = json_str.replace('"', "'")
    json_str = json_str.replace("'from'", '"from"')
    json_str = json_str.replace("'human'", '"human"')
    json_str = json_str.replace("'gpt'", '"gpt"')
    json_str = json_str.replace("'value':", '"value":"')
    json_str = json_str.replace("}", '"},')
    json_str = json_str.replace("},]", '}]')
    json_str = re.sub(r'\\', '', json_str)
    json_str = json_str.replace(": '", ': "')
    json_str = json_str.replace("'}", '"}')
    json_str = json_str.lower()
    json_str = json_str.replace("charlie", '')
    json_str = json_str.replace("alex", '')
    return json_str

# Load data in chunks
def load_conversation_data(df, chunksize=1000):
    conversations = []
    for conv in df['conversations']:
        fixed_conv = fix_json_string(conv)
        try:
            conversations.append(json.loads(fixed_conv))
        except:
            continue
    return conversations

# Function to create conversation history
def create_conversation_history(conversations):
    inputs, targets,only_inputs = [], [],[]
    for conv in conversations:
        temp = []
        for turn in conv:
            if turn.get('from') == 'human':
                only_inputs.append(turn.get('value', ''))
                inputs.append(" ".join(temp) + " " + turn.get('value', ''))
                temp.append(turn.get('value', ''))
            elif turn.get('from') == 'gpt':
                targets.append(turn.get('value', ''))
                temp.append(turn.get('value', ''))
    return inputs, targets, only_inputs


# Load data
df = pd.read_csv('/content/chat_data - Copy.csv')
df = df.dropna()
conversations = load_conversation_data(df)
print("No. of conversations: ",len(conversations))
