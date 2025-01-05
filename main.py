import streamlit as st
import torch
from transformers import ConvNextImageProcessor, ConvNextForImageClassification
import cv2
import numpy as np
from PIL import Image
import os
import random


# 加载训练好的模型和对应的图像处理器
model_name_or_path = "ThyroidTumorClassification"
feature_extractor = ConvNextImageProcessor.from_pretrained(model_name_or_path)
model = ConvNextForImageClassification.from_pretrained(model_name_or_path)
model.eval()  # 设置模型为评估模式


def preprocess_image(image):
    """
    对输入的图像进行预处理，使其符合模型输入要求
    """
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 转换颜色通道顺序（如果需要）
    img = feature_extractor(images=img, return_tensors="pt")["pixel_values"]
    return img


def predict(image):
    """
    使用加载的模型对输入图像进行预测并返回预测结果
    """
    with torch.no_grad():
        inputs = preprocess_image(image)
        outputs = model(inputs)
        logits = outputs.logits
        predicted_class_idx = torch.argmax(logits, dim=1).item()
    return predicted_class_idx

def get_random_local_image():
    local_images_folder = "thyroid"  # 存放本地图像的文件夹，需根据实际情况修改
    if os.path.exists(local_images_folder) and os.path.isdir(local_images_folder):
        jpg_files = [f for f in os.listdir(local_images_folder) if f.endswith('.jpg')]
        if jpg_files:
            random_image_path = os.path.join(local_images_folder, random.choice(jpg_files))
            return Image.open(random_image_path).convert("RGB")
    return None

def main():
    st.title("甲状腺肿瘤分类模型推理:1为异常,0为正常！")
    # 选项：从本地随机选择图像 或 上传图像
    option = st.radio("选择图像来源", ("从本地随机选择", "上传图像"))
    if option == "从本地随机选择":
        local_image = get_random_local_image()
        if local_image is not None:
            st.image(local_image, caption="本地随机图像", use_container_width=True)
            if st.button("对本地图像进行预测"):
                prediction = predict(local_image)
                st.write(f"预测结果：{prediction}")
    else:
        uploaded_file = st.file_uploader("请上传甲状腺超声图像", type=["jpg", "png", "jpeg"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="上传的图像", use_container_width=True)
            if st.button("进行预测"):
                prediction = predict(image)
                st.write(f"预测结果：{prediction}")


if __name__ == "__main__":
    main()