import os
import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
from sklearn.model_selection import train_test_split

# SỬA ĐƯỜNG DẪN NÀY: Trỏ đến thư mục Animals-10 trên máy bạn
path_data = r'C:\Users\SAM\OneDrive\Máy tính\Ai_Captcha\data\Animals-10' 

# Lấy danh sách label
labels = sorted([d for d in os.listdir(path_data) if os.path.isdir(os.path.join(path_data, d))])
print("Các loài vật phát hiện được:", labels)





data = []
for label in labels:
    folder_path = os.path.join(path_data, label)
    for img_file in os.listdir(folder_path):
        if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
            img_path = os.path.join(folder_path, img_file)
            data.append((img_path, label))

df = pd.DataFrame(data, columns=['filepath', 'label'])
train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['label'], random_state=42)

print(f"Tổng số ảnh: {len(df)}")
print(f"Train: {len(train_df)} - Test: {len(test_df)}")





def image_dataset_from_dataframe(df, image_size=(64, 64), batch_size=32, shuffle=True):
    label_names = sorted(df['label'].unique())
    label_to_index = {name: index for index, name in enumerate(label_names)}
    
    paths = df['filepath'].values
    labels_idx = df['label'].map(label_to_index).values

    def load_image(path, label):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.image.resize(image, image_size)
        image = tf.cast(image, tf.float32) / 255.0
        return image, tf.one_hot(label, depth=len(label_names))

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_idx))
    if shuffle: ds = ds.shuffle(len(df))
    ds = ds.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
    return ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

train_ds = image_dataset_from_dataframe(train_df)
test_ds = image_dataset_from_dataframe(test_df, shuffle=False)





from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.regularizers import l2

base_model = MobileNetV2(input_shape=(64, 64, 3), include_top=False, weights='imagenet')
base_model.trainable = False

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(64, activation='relu', kernel_regularizer=l2(0.01))(x)
x = Dropout(0.5)(x)
predictions = Dense(len(labels), activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

print("Bắt đầu huấn luyện CNN...")
history = model.fit(train_ds, validation_data=test_ds, epochs=5)

# LƯU FILE ĐỂ LÀM BACKEND
model.save("captcha_animal_model.h5")
print("Đã lưu model thành file captcha_animal_model.h5")





from sklearn.decomposition import IncrementalPCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# 1. Chạy PCA
ipca = IncrementalPCA(n_components=32)
for images, _ in train_ds:
    batch_flat = tf.reshape(images, (images.shape[0], -1)).numpy()
    if batch_flat.shape[0] >= ipca.n_components:
        ipca.partial_fit(batch_flat)

def extract_pca(ds):
    X, y = [], []
    for imgs, lbls in ds:
        flat = tf.reshape(imgs, (imgs.shape[0], -1)).numpy()
        X.append(ipca.transform(flat))
        y.append(np.argmax(lbls.numpy(), axis=1))
    return np.concatenate(X), np.concatenate(y)

X_train_pca, y_train_pca = extract_pca(train_ds)
X_test_pca, y_test_pca = extract_pca(test_ds)

# 2. Train KNN & RF
knn = KNeighborsClassifier(n_neighbors=5).fit(X_train_pca, y_train_pca)
rf = RandomForestClassifier(n_estimators=100, max_depth=10).fit(X_train_pca, y_train_pca)

print(f"KNN Acc: {accuracy_score(y_test_pca, knn.predict(X_test_pca)):.4f}")
print(f"RF Acc: {accuracy_score(y_test_pca, rf.predict(X_test_pca)):.4f}")
print(f"CNN Acc: {model.evaluate(test_ds, verbose=0)[1]:.4f}")





for images, labels_batch in test_ds.take(1):
    preds = model.predict(images)
    plt.figure(figsize=(15, 8))
    for i in range(10): # Hiển thị 10 ảnh mẫu
        plt.subplot(2, 5, i+1)
        plt.imshow(images[i].numpy())
        p_idx, t_idx = np.argmax(preds[i]), np.argmax(labels_batch[i])
        color = 'green' if p_idx == t_idx else 'red'
        plt.title(f"P: {labels[p_idx]}\nT: {labels[t_idx]}", color=color)
        plt.axis('off')
    plt.show()
