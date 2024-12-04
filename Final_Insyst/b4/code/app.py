import datetime
from flask import Flask, render_template, Response, jsonify, request
import cv2
import numpy as np
from face_recognition import get_face_embedding, match_face, load_embeddings, save_embeddings
import os
from werkzeug.utils import secure_filename
from flask_cors import CORS
import time
import pymysql  # Sử dụng PyMySQL thay cho flask_mysqldb

app = Flask(__name__)
CORS(app)

# MySQL configuration (Sử dụng PyMySQL thay cho flask_mysqldb)
app.config['MYSQL_HOST'] = 'localhost'  
app.config['MYSQL_USER'] = 'root'  
app.config['MYSQL_PASSWORD'] = 'hvxk2003'  
app.config['MYSQL_DB'] = 'httm1'  

# Load embeddings and names
face_embeddings, names = load_embeddings()

# Video capture setup
cap = cv2.VideoCapture(0)

# Global list to store recognized faces and their detection time
recognized_faces = []  # List of dictionaries with 'name' and 'time'

def generate_frames():
    global recognized_faces
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Resize frame for faster processing
            small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            rgb_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

            # Detect and recognize face
            embedding = get_face_embedding(rgb_frame)
            if embedding is not None:
                name = match_face(embedding, face_embeddings, names)
                if name:
                    # Check if face is already recognized
                    if not any(face['name'] == name for face in recognized_faces):
                        # Add new face with timestamp
                        recognized_faces.append({
                            'name': name,
                            'time': time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())
                        })
                        # Save detection to MySQL
                    log_detection(name)

            # Encode the frame for streaming
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

def log_detection(name):
    try:
        # Sử dụng pymysql.connect thay vì mysql.connect
        conn = pymysql.connect(
            host=app.config['MYSQL_HOST'],
            user=app.config['MYSQL_USER'],
            password=app.config['MYSQL_PASSWORD'],
            database=app.config['MYSQL_DB']
        )
        cursor = conn.cursor()

        # Insert the detected face into the database
        query = "INSERT INTO detection_history (name, time_detected) VALUES (%s, %s)"
        cursor.execute(query, (name, time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())))

        conn.commit()
        cursor.close()
        conn.close()
    except Exception as e:
        print(f"Error: {e}")

@app.route('/get_recognized_faces', methods=['GET'])
def get_recognized_faces():
    return jsonify({'faces': recognized_faces}), 200

@app.route('/')
def index():
    return render_template('index.html')

# Add face to the system
UPLOAD_FOLDER = 'static/faces_folder'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/add_face', methods=['POST'])
def add_face():
    try:
        if 'face_image' not in request.files or 'name' not in request.form:
            return "Invalid request. Please provide 'face_image' and 'name'.", 400

        file = request.files['face_image']
        name = request.form['name']

        if file.filename == '':
            return "No file selected.", 400

        # Lưu ảnh vào thư mục
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        filepath = os.path.join(UPLOAD_FOLDER, f"{name}.jpg")
        file.save(filepath)

        # Tính toán embedding cho khuôn mặt
        embedding = get_face_embedding(filepath)
        if embedding is None:
            return "No face detected in the image.", 400

        # Lưu embedding và tên
        face_embeddings, names = load_embeddings()
        face_embeddings = np.append(face_embeddings, [embedding], axis=0)  # Thêm embedding mới vào cuối mảng
        names = np.append(names, [name])  # Thêm tên vào cuối mảng

        # Lưu vào file
        save_embeddings(face_embeddings, names)

        return "Face added and embedding saved successfully."

    except Exception as e:
        print(f"Error: {e}")
        return "An error occurred while adding the face.", 500

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/add_face_page')
def add_face_page():
    return render_template('add_face.html')



from datetime import datetime

@app.route('/get_detection_history', methods=['GET'])
def get_detection_history():
    # Lấy thông tin ngày giờ bắt đầu và kết thúc từ form
    start_datetime = request.args.get('start_datetime')
    end_datetime = request.args.get('end_datetime')

    # Kiểm tra nếu có giá trị và chuyển đổi chúng thành datetime object
    if start_datetime:
        start_datetime = datetime.strptime(start_datetime, '%Y-%m-%dT%H:%M')
    if end_datetime:
        end_datetime = datetime.strptime(end_datetime, '%Y-%m-%dT%H:%M')

    conn = pymysql.connect(
        host=app.config['MYSQL_HOST'],
        user=app.config['MYSQL_USER'],
        password=app.config['MYSQL_PASSWORD'],
        database=app.config['MYSQL_DB']
    )
    cursor = conn.cursor()

    # Nếu không có ngày giờ bắt đầu và kết thúc, hiển thị tất cả các bản ghi
    if not start_datetime or not end_datetime:
        query = "SELECT * FROM detection_history ORDER BY time_detected DESC"
        cursor.execute(query)
    else:
        # Nếu có chọn khoảng thời gian, sử dụng điều kiện BETWEEN
        query = "SELECT * FROM detection_history WHERE time_detected BETWEEN %s AND %s ORDER BY time_detected DESC"
        cursor.execute(query, (start_datetime, end_datetime))

    rows = cursor.fetchall()

    history = []
    for row in rows:
        history.append({'id': row[0], 'name': row[1], 'time_detected': row[2]})

    conn.close()

    return render_template('history.html', history=history)



if __name__ == '__main__':
    app.run(debug=True)
