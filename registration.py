import cv2
import numpy as np
from numpy import asarray
import os
from mtcnn.mtcnn import MTCNN
from PIL import Image
from matplotlib import pyplot as plt

face_cascade = cv2.CascadeClassifier("models/haarcascade_frontalface_default.xml")
root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir,'dataset')

#assign the MTCNN detector
detector = MTCNN()

#Method to extract Face
# def extract_image(image):
#   img1 = Image.open(image)            #open the image
#   img1 = img1.convert('RGB')          #convert the image to RGB format 
#   pixels = asarray(img1)              #convert the image to numpy array
                    
#   f = detector.detect_faces(pixels)
#   #fetching the (x,y)co-ordinate and (width-->w, height-->h) of the image
#   x1,y1,w,h = f[0]['box']             
#   x1, y1 = abs(x1), abs(y1)
#   x2 = abs(x1+w)
#   y2 = abs(y1+h)
#   #locate the co-ordinates of face in the image
#   store_face = pixels[y1:y2,x1:x2]
#   plt.imshow(store_face)
#   image1 = Image.fromarray(store_face,'RGB')    #convert the numpy array to object
#   image1 = image1.resize((160,160))             #resize the image
#   face_array = asarray(image1)                  #image to array
#   return face_array

name = input("Enter your name: ")
input_directory = os.path.join(dataset_dir,name)
if not os.path.exists(input_directory):
    os.makedirs(input_directory, exist_ok = 'True')
    count = 1
    print("[INFO] starting video stream...")
    video_capture = cv2.VideoCapture(0)
    while count <= 50:
        try:
            check, frame = video_capture.read()
            gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray,1.3,5)
            #faces = detector.detect_faces(gray,1.3,5)
            for (x,y,w,h) in faces:  
                face = frame[y-5:y+h+5,x-5:x+w+5]
                resized_face = cv2.resize(face,(160,160))
                cv2.imwrite(os.path.join(input_directory,name + str(count) + '.jpg'),resized_face)
                cv2.rectangle(frame, (x,y), (x+w, y+h),(0,0,255), 2)
                count += 1
            # show the output frame
            cv2.imshow("Frame",frame)
            key = cv2.waitKey(1)
            if key == ord('q'):
                break
        except Exception as e:
            pass
    video_capture.release()
    cv2.destroyAllWindows()
else:
    print("Photo already added for this user..Please provide another name for storing datasets")