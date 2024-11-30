import numpy as np
import cv2
import os
import torch
from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1
from scipy.spatial.distance import cosine

# Khởi tạo MTCNN và InceptionResnetV1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device=device)  
resnet = InceptionResnetV1(pretrained='vggface2').eval().to(device)

# Hàm chuẩn hóa ảnh
def normalize_image(image):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_resized = cv2.resize(image_rgb, (160, 160))
    
  
    image_normalized = image_resized / 255.0

    return np.expand_dims(image_normalized, axis=0)

# Hàm tính toán embedding của khuôn mặt từ MTCNN và InceptionResnetV1
def get_face_embedding(image):
    if isinstance(image, str):
        image = cv2.imread(image)

    image_normalized = normalize_image(image)
    pil_image = Image.fromarray((image_normalized[0] * 255).astype(np.uint8))  

    # Phát hiện khuôn mặt
    faces = mtcnn(pil_image)  
    if faces is None:
        return None  

    embeddings = []
    for face in faces: 
        face_embedding = resnet(face.unsqueeze(0).to(device)).detach().cpu().numpy()  
        embeddings.append(face_embedding[0])  

    return embeddings[0] if embeddings else None  

# Hàm so sánh và tìm kiếm khuôn mặt
def match_face(embedding, face_embeddings, names):
    distances = [cosine(embedding, emb) for emb in face_embeddings]  # Tính khoảng cách cosine
    min_distance_idx = np.argmin(distances)
    if distances[min_distance_idx] < 0.5:  # Ngưỡng chấp nhận độ tương đồng
        return names[min_distance_idx]
    return None

def load_embeddings():
    if os.path.exists('face_embeddings_facenet.npy') and os.path.exists('names_facenet.npy'):
        face_embeddings = np.load('face_embeddings_facenet.npy', allow_pickle=True)
        names = np.load('names_facenet.npy', allow_pickle=True)
    else:
        face_embeddings, names = [], []
    return face_embeddings, names

# Lưu face embeddings và tên
def save_embeddings(face_embeddings, names):
    np.save('face_embeddings_facenet.npy', face_embeddings)
    np.save('names_facenet.npy', names)
