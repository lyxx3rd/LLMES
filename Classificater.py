from flask import Flask, request, jsonify
import torch
import threading
from transformers import AutoTokenizer
import time

app = Flask(__name__)

# 加载模型和分词器
model_path = "../Model/dienstag/chinese-roberta-wwm-ext"
classify_model = torch.load("../Model/Model_saved/classify_model.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
classify_model.to(DEVICE)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# 标记模型是否已加载完成
model_loaded = False

def load_model():
    global model_loaded
    # 模拟模型加载过程
    time.sleep(10)  # 假设模型加载需要10秒
    model_loaded = True

# 启动模型加载线程
model_loading_thread = threading.Thread(target=load_model)
model_loading_thread.start()

def predict_single_sentence(sentence: str):
    # Tokenize the sentence with the same settings as during training
    inputs = tokenizer(sentence, return_tensors="pt", max_length=256, padding=True, truncation=True)
    # Move the input tensors to the correct device
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}
    
    # Set the model to evaluation mode (important if your model has layers like dropout or batchnorm)
    classify_model.eval()

    # Perform inference
    with torch.no_grad():
        outputs = classify_model(**inputs)
    
    # Get the predicted class. This assumes that you're using a classification model
    # and that the model returns logits.
    # You might need to modify this depending on what your model's forward pass returns
    _, predicted = torch.max(outputs.logits, 1)
    
    return predicted.item()  # Convert the tensor to a Python scalar

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    sentence = data.get('sentence', '')
    
    if not sentence:
        return jsonify({'error': 'No sentence provided'}), 400
    
    try:
        prediction = predict_single_sentence(sentence)
        return jsonify({'prediction': prediction})
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/health', methods=['GET'])
def health_check():
    if model_loaded:
        return jsonify({'status': 'OK'}), 200
    else:
        return jsonify({'status': 'Loading'}), 503

def start_flask_app():
    app.run(debug=False, port=5000)
    
if __name__ == '__main__':
    start_flask_app()