from flask import Blueprint, send_file, render_template,flash, request, redirect, json,session, Response
from flask_session import Session
import cv2
from datetime import datetime

import numpy as np

views = Blueprint('views', __name__)

camera = cv2.VideoCapture(0)
model = 'website/data/best.onnx'
text  = "website/data/kanker_text.txt"
net = cv2.dnn.readNetFromONNX(model)
file = open(text, "r")
classess = file.read().split('\n')# generate frame by frame from camera


def camera_frames():
    while True:
        # Capture frame-by-frame
        success, frame = camera.read()  # read the camera frame
        if not success:
            break
        else:
            frame = cv2.resize(frame, (1000, 600))
            blob = cv2.dnn.blobFromImage(frame, scalefactor=1.0 / 255, size=(640, 640), mean=[0, 0, 0], swapRB=True, crop=False)
            net.setInput(blob)
            detections = net.forward()[0]
            classes_ids = []
            confidences = []
            boxes = []
            rows = detections.shape[0]

            img_width, img_height = frame.shape[1], frame.shape[0]
            x_scale = img_width/640
            y_scale = img_height/640

            for i in range(rows):
                row = detections[i]
                confidence = row[4]
                if confidence > 0.5:
                    classes_score = row[5:]
                    ind = np.argmax(classes_score)
                    if classes_score[ind] > 0.5:
                        classes_ids.append(ind)
                        confidences.append(confidence)
                        cx, cy, w, h = row[:4]
                        x1 = int((cx - w / 2) * x_scale)
                        y1 = int((cy - h / 2) * y_scale)
                        width = int(w * x_scale)
                        height = int(h * y_scale)
                        box = np.array([x1, y1, width, height])
                        boxes.append(box)
            indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.5)
            for i in indices:
                x1, y1, w, h = boxes[i]
                label = classess[classes_ids[i]]
                conf = confidences[i]
                text = label
                cv2.rectangle(frame, (x1, y1), (x1 + w, y1 + h), (255, 0, 0), 2)
                cv2.putText(frame, text, (x1, y1 - 2), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 255), 2)
            if type(indices) is not tuple:
                cv2.imwrite('website/static/hasil_foto_scan.jpg', frame)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@views.route('/ambil-foto')
def ambilFoto():
    return send_file('static/foto.jpg', as_attachment=True)

@views.route('/video')
def video():
    #Video streaming route. Put this in the src attribute of an img tag
    return Response(camera_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@views.route('/')
def home():
    return render_template('index.html', ttl='Deteksi Kanker Melanoma')

@views.route('/detect')
def detect():
    return render_template('camera_detection.html', ttl='Deteksi Kanker Melanoma')
@views.route('/information')
def info():
    return render_template('information.html', ttl='Deteksi Kanker Melanoma')

@views.route('/superficial-melanoma')
def superficialMelanoma():
    return render_template('superficial-melanoma.html', ttl='Superficial Melanoma')

@views.route('/nodular-melanoma')
def nodularMelanoma():
    return render_template('nodular-melanoma.html', ttl='Nodular Melanoma')

@views.route('/lentigo-melanoma')
def lentigoMelanoma():
    return render_template('lentigo-melanoma.html', ttl='Lentigo Maligna Melanoma')

@views.route('/acral-melanoma')
def acralMelanoma():
    return render_template('acral-melanoma.html', ttl='Acral Lentiginous Melanoma')

@views.route('/faktor-terkena-melanoma')
def faktorMelanoma():
    return render_template('faktor-melanoma.html', ttl='Faktor yang Dapat Menyebabkan Terkena Kanker Melanoma')