from flask import Flask, Response, render_template, redirect, url_for, request
import face_recognition
import cv2
import numpy as np
import os
import os.path
import pickle
from PIL import Image, ImageDraw

root_dir = os.getcwd()
models_dir = os.path.join(root_dir, 'models')

model_path = os.path.join(models_dir, 'knn.pkl')

app = Flask(__name__)

camera = cv2.VideoCapture(0)

def predict(X_frame, knn_clf=None, model_path=None, distance_threshold=0.4):
    
    if knn_clf is None and model_path is None:
        raise Exception("Must supply knn classifier either thourgh knn_clf or model_path")

    # Load a trained KNN model (if one was passed in)
    if knn_clf is None:
        with open(model_path, 'rb') as f:
            knn_clf = pickle.load(f)

    X_face_locations = face_recognition.face_locations(X_frame)

    # If no faces are found in the image, return an empty result.
    if len(X_face_locations) == 0:
        return []

    # Find encodings for faces in the test image
    faces_encodings = face_recognition.face_encodings(X_frame, known_face_locations=X_face_locations)

    # Use the KNN model to find the best matches for the test face
    closest_distances = knn_clf.kneighbors(faces_encodings, n_neighbors=1)
    are_matches = [closest_distances[0][i][0] <= distance_threshold for i in range(len(X_face_locations))]

    # Predict classes and remove classifications that aren't within the threshold
    return [(pred, loc) if rec else ("unknown", loc) for pred, loc, rec in zip(knn_clf.predict(faces_encodings), X_face_locations, are_matches)]


def show_prediction_labels_on_image(frame, predictions):
    
    pil_image = Image.fromarray(frame)
    draw = ImageDraw.Draw(pil_image)

    for name, (top, right, bottom, left) in predictions:
        # enlarge the predictions for the full sized image.
        top *= 2
        right *= 2
        bottom *= 2
        left *= 2
        # Draw a box around the face using the Pillow module
        draw.rectangle(((left, top), (right, bottom)), outline=(0, 0, 255))

        # There's a bug in Pillow where it blows up with non-UTF-8 text
        # when using the default bitmap font
        name = name.encode("UTF-8")

        # Draw a label with a name below the face
        text_width, text_height = draw.textsize(name)
        draw.rectangle(((left, bottom - text_height - 10), (right, bottom)), fill=(0, 0, 255), outline=(0, 0, 255))
        draw.text((left + 6, bottom - text_height - 5), name, fill=(255, 255, 255, 255))

    # Remove the drawing library from memory as per the Pillow docs.
    del draw
    # Save image in open-cv format to be able to show it.

    opencvimage = np.array(pil_image)
    return opencvimage

def gen_frames(option=None):  
    print(option)
    while True:
        success, frame = camera.read()  # read the camera frame

        if not success:
            break
        else:

            if option is not None: 
                if option == 'face_rec':
                    process_this_frame = 29
                    img = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
                    process_this_frame = process_this_frame + 1
                    if process_this_frame % 30 == 0:
                        predictions = predict(img, model_path=model_path)
                    frame = show_prediction_labels_on_image(frame, predictions)
            
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

@app.route('/')
def index():
    # return 'Hello World'
    return render_template('web.html')

@app.route('/video_feed/<option>')
def video_feed(option):
    return Response(gen_frames(option), mimetype='multipart/x-mixed-replace; boundary=frame')    

# @app.route('/face_rec')
# def face_rec():
#     return redirect(url_for('video_feed',option = 'face_rec'))
    # return Response(gen_frames(option='face_rec'), mimetype='multipart/x-mixed-replace; boundary=frame') 

if __name__ == '__main__':
    app.debug = True
    # app.run()
    app.run(host="0.0.0.0")