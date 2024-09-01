from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from flask_cors import CORS
from PIL import Image
import io

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": ["https://storied-kangaroo-bf7b34.netlify.app","https://splendid-moonbeam-252fc3.netlify.app"]}})

def skin_tone_to_color(skin_tone_value):
    if skin_tone_value < 80:
        color_rgb = (45, 43, 31)  # Dark brown
        color_name = "Dark Brown"
    elif skin_tone_value < 120:
        color_rgb = (120, 102, 77)  # Medium brown
        color_name = "Medium Brown"
    elif skin_tone_value < 150:
        color_rgb = (194, 164, 128)  # Light brown
        color_name = "Light Brown"
    elif skin_tone_value < 180:
        color_rgb = (255, 204, 153)  # Fair skin
        color_name = "Fair Skin"
    else:
        color_rgb = (255, 255, 255)  # Very fair skin
        color_name = "Very Fair Skin"

    color_hex = '#{:02x}{:02x}{:02x}'.format(*color_rgb)

    return color_hex, color_name

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > conf_threshold:
            x1 = int(detections[0, 0, i, 3] * frameWidth)
            y1 = int(detections[0, 0, i, 4] * frameHeight)
            x2 = int(detections[0, 0, i, 5] * frameWidth)
            y2 = int(detections[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frameOpencvDnn, (x1, y1), (x2, y2), (0, 255, 0), int(round(frameHeight / 150)), 8)
    return frameOpencvDnn, faceBoxes

def analyze_face(img):
    faceProto = "opencv_face_detector.pbtxt"
    faceModel = "opencv_face_detector_uint8.pb"
    ageProto = "age_deploy.prototxt"
    ageModel = "age_net.caffemodel"
    genderProto = "gender_deploy.prototxt"
    genderModel = "gender_net.caffemodel"

    MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
    ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
    genderList = ['Male', 'Female']

    faceNet = cv2.dnn.readNet(faceModel, faceProto)
    ageNet = cv2.dnn.readNet(ageModel, ageProto)
    genderNet = cv2.dnn.readNet(genderModel, genderProto)

    frame = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    padding = 20
    resultImg, faceBoxes = highlightFace(faceNet, frame)
    results = []

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1] - padding): min(faceBox[3] + padding, frame.shape[0] - 1),
                     max(0, faceBox[0] - padding): min(faceBox[2] + padding, frame.shape[1] - 1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        gender = genderList[genderPreds[0].argmax()]

        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        age = ageList[agePreds[0].argmax()]

        skin_tone = np.mean(cv2.cvtColor(face, cv2.COLOR_BGR2YCrCb)[:, :, 0])
        color_hex, color_name = skin_tone_to_color(skin_tone)

        results.append({
            'gender': gender,
            'age': age[1:-1] + ' years',
            'skin_tone': skin_tone,
            'color_name': color_name,
            'color_hex': color_hex
        })

    return results

@app.route('/detect_skin_tone', methods=['POST'])
def detect_skin_tone():
    if 'image' not in request.files:
        return jsonify({'error': 'No image file'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    try:
        img = Image.open(file.stream)
        img_ip = np.array(img)
    except Exception as e:
        return jsonify({'error': f'Image processing error: {str(e)}'}), 500

    try:
        results = analyze_face(img_ip)
        if results:
            return jsonify(results[0])  # Return the first result
        else:
            return jsonify({'error': 'No faces detected'}), 404
    except Exception as e:
        return jsonify({'error': f'Analysis error: {str(e)}'}), 500

@app.errorhandler(404)
def page_not_found(e):
    return jsonify({'error': 'Not Found'}), 404

if __name__ == '__main__':
    app.run(debug=True)
