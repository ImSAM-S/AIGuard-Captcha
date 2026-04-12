from flask import Flask, render_template, request, jsonify
import tensorflow as tf
import numpy as np
import os
import random
from PIL import Image
import shutil

app = Flask(__name__)

# 1. Cấu hình đường dẫn và Model
# Đảm bảo tên file khớp với file thực tế của bạn
MODEL_PATH = 'captcha_animal_model.h5'
PATH_DATA = r'C:\Users\SAM\OneDrive\Máy tính\Ai_Captcha\data\Animals-10'

# Nạp model AI
model = tf.keras.models.load_model(MODEL_PATH)
labels = sorted(['butterfly', 'cat', 'chicken', 'cow', 'dog', 'elephant', 'horse', 'bird', 'spider', 'bee_imgs'])

# Tạo thư mục tạm để chứa ảnh hiển thị lên web
TEMP_DIR = os.path.join('static', 'temp')
if not os.path.exists(TEMP_DIR):
    os.makedirs(TEMP_DIR)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/get_captcha', methods=['GET'])
def get_captcha():
    # Bước 1: Chọn loài vật mục tiêu (câu hỏi)
    target_label = random.choice(labels)
    
    # Bước 2: Lấy 1 ảnh ĐÚNG
    correct_folder = os.path.join(PATH_DATA, target_label)
    correct_img_name = random.choice(os.listdir(correct_folder))
    correct_img_path = os.path.join(correct_folder, correct_img_name)
    
    # Bước 3: Lấy 8 ảnh SAI từ các loài khác
    other_labels = [l for l in labels if l != target_label]
    choices = [{"path": correct_img_path, "label": target_label}]
    
    for i in range(8):
        wrong_label = random.choice(other_labels)
        wrong_folder = os.path.join(PATH_DATA, wrong_label)
        wrong_img_name = random.choice(os.listdir(wrong_folder))
        choices.append({
            "path": os.path.join(wrong_folder, wrong_img_name),
            "label": wrong_label
        })
    
    # Trộn ngẫu nhiên vị trí 9 ảnh
    random.shuffle(choices)
    
    # Bước 4: Lưu 9 ảnh vào thư mục static/temp để Web hiển thị
    grid_data = []
    for i, item in enumerate(choices):
        img_filename = f"captcha_{i}.jpg"
        save_path = os.path.join(TEMP_DIR, img_filename)
        
        # Resize ảnh về 150x150 để hiển thị lưới cho đẹp
        img = Image.open(item['path']).convert('RGB').resize((150, 150))
        img.save(save_path)
        
        grid_data.append({
            "id": i,
            "url": f"/static/temp/{img_filename}",
            "label": item['label'] # Nhãn này dùng để đối chiếu khi click
        })

    return jsonify({
        "question": f"Hãy chọn hình có con: {target_label}",
        "target": target_label,
        "images": grid_data
    })

if __name__ == '__main__':
    app.run(debug=True)