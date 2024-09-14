from flask import Flask, request, jsonify, render_template
import torch
from transformers import BertTokenizer, EncoderDecoderModel

app = Flask(__name__)

# Define the paths to your saved model and tokenizer
model_path = 'results2final2'
tokenizer_path = 'results2final2'

# Load the model and tokenizer
model2 = EncoderDecoderModel.from_pretrained(model_path)
tokenizer2 = BertTokenizer.from_pretrained(tokenizer_path)

# Ensure the model is on the right device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2.to(device)

def chat_with_model(input_text, model, tokenizer):
    model.eval()

    inputs = tokenizer(input_text, return_tensors='pt', padding=True, truncation=True)
    input_ids = inputs['input_ids'].to(model.device)
    attention_mask = inputs['attention_mask'].to(model.device)

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

    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return response

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.json['message']
    response = chat_with_model(user_input, model2, tokenizer2)
    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(debug=True)
