import streamlit as st
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
import tensorflow as tf
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt

# Tải mô hình YOLO
model_yolo = YOLO("best.pt")

# Tải mô hình CNN
model_cnn = tf.keras.models.load_model('cnn_model.h5')
feature_extractor = tf.keras.Model(inputs=model_cnn.layers[0].input, outputs=model_cnn.layers[-7].output)

# Đọc ý nghĩa biển báo từ name_signs.txt
with open("name_signs.txt", "r", encoding='utf-8') as f:
    name_signs = [line.strip() for line in f.readlines()]

# Đọc vào file csv chứa các đặc trưng đã lưu
feature_signs = pd.read_csv("features.csv")

# Preprocess image for CNN model
def preprocess_image(image):
    img = cv2.resize(image, (32, 32)) / 255.0
    img = img.reshape(1, 32, 32, 3)
    return img

def regconize_traffic_sign(feature):
    similarities = []
    for i in range(0, feature_signs.shape[0]):
        temp = np.array(feature_signs.iloc[i, 1:]).reshape(1, -1)
        simi = cosine_similarity(feature, temp)
        similarities.append(simi)
    # st.write(similarities)
    index = int(feature_signs.iloc[np.argmax(similarities), 0])
    return index

# Hàm detect biển báo giao thông
def detect_traffic_sign(image, conf):
    with st.spinner("Đang nhận diện biển báo giao thông..."):
        results = model_yolo.predict(image, conf=conf)
        st.write("Ảnh kết quả detect biển báo giao thông")
        st.image(results[0].plot(conf=True, labels=False))

        cols = st.columns(2)
        for result in results:
            orig_img = result.orig_img
            boxes = result.boxes

            for i, box in enumerate(boxes):
                x_min, y_min, x_max, y_max = map(int, box.xyxy[0])
                cropped_img = orig_img[y_min:y_max, x_min:x_max]  # Crop biển báo khỏi ảnh gốc

                with cols[i % 2]:
                    st.write(f"**Biển báo {i+1}**")
                    st.image(cropped_img)

                    cropped_img = preprocess_image(cropped_img)
                    feature = feature_extractor.predict(cropped_img)

                    index = regconize_traffic_sign(feature)
                    st.write(f"**Tên biển báo:** {name_signs[index - 1]}")

# Giao diện chính
st.set_page_config(page_title="Nhận diện biển báo giao thông", layout="wide")
st.title("🚦 Ứng dụng nhận diện biển báo giao thông")
st.markdown("""
Ứng dụng này sử dụng YOLO và CNN để phát hiện và nhận diện các biển báo giao thông trong ảnh.
- **Bước 1**: Tải ảnh biển báo giao thông lên.
- **Bước 2**: Chọn mức độ `confidence`.
- **Bước 3**: Xem kết quả!
""")

st.sidebar.header("Cấu hình")
conf = st.sidebar.slider("Ngưỡng confidence", 0.0, 1.0, 0.5)
uploaded_file = st.sidebar.file_uploader("Chọn file ảnh", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Đọc ảnh
    image = cv2.imdecode(np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8), 1)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    st.image(image, caption="Ảnh gốc")
    detect_traffic_sign(image, conf)
else:
    st.info("Vui lòng tải lên một file ảnh để bắt đầu.")
