import math
from sklearn import neighbors
import os
import os.path
import pickle
import numpy as np
# from PIL import Image, ImageDraw
import face_recognition
from face_recognition.face_recognition_cli import image_files_in_folder

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

root_dir = os.getcwd()
dataset_dir = os.path.join(root_dir, 'dataset')
models_dir = os.path.join(root_dir, 'models')

#train_dir = os.listdir(dataset_dir)

save_model_path = os.path.join(models_dir, 'knn.pkl')

data_model_path = os.path.join(models_dir, 'data.pkl')

def storeData(data, file):
    dbfile = open(file, 'wb') 
    # print(data)     
    # source, destination    
    pickle.dump(data, dbfile)                     
    dbfile.close()

def loadData(file):
    db = [[],[]]
    if os.path.exists(file):
        # dbfile = open(file, 'rb') 
        with open(file,'rb') as rfp:     
            db = pickle.load(rfp)       
        # for keys in db:
        #     print(keys, '=>', db[keys])
            rfp.close()  
    # print(db)        
    return db  

def train(train_dir, train_id, model_save_path=None, n_neighbors=None, knn_algo='ball_tree', verbose=False):
    """
    Trains a k-nearest neighbors classifier for face recognition.

    :param train_dir: directory that contains a sub-directory for each known person, with its name.

     (View in source code to see train_dir example tree structure)

     Structure:
        <train_dir>/
        ├── <person1>/
        │   ├── <somename1>.jpeg
        │   ├── <somename2>.jpeg
        │   ├── ...
        ├── <person2>/
        │   ├── <somename1>.jpeg
        │   └── <somename2>.jpeg
        └── ...

    :param model_save_path: (optional) path to save model on disk
    :param n_neighbors: (optional) number of neighbors to weigh in classification. Chosen automatically if not specified
    :param knn_algo: (optional) underlying data structure to support knn.default is ball_tree
    :param verbose: verbosity of training
    :return: returns knn classifier that was trained on the given data.
    """

    """
    Open existing data if exist
    """

    data = loadData(data_model_path)

    # if os.path.exists(data_model_path):
    #     # "with" statements are very handy for opening files. 
    #     with open(data_model_path,'rb') as rfp: 
    #         data = pickle.load(rfp)
    # print(data)
    X = []
    y = []

    # Loop through each person in the training set
    # for class_dir in os.listdir(train_dir):
    #     if not os.path.isdir(os.path.join(train_dir, class_dir)):
    #         continue

        # Loop through each training image for the current person
    for img_path in image_files_in_folder(train_dir):
        image = face_recognition.load_image_file(img_path)
        face_bounding_boxes = face_recognition.face_locations(image)

        if len(face_bounding_boxes) != 1:
            # If there are no people (or too many people) in a training image, skip the image.
            if verbose:
                print("Image {} not suitable for training: {}".format(img_path, "Didn't find a face" if len(face_bounding_boxes) < 1 else "Found more than one face"))
        else:
            # Add face encoding for current image to the training set
            X.append(face_recognition.face_encodings(image, known_face_locations=face_bounding_boxes)[0])
            y.append(train_id)

    # print(X)
    # print(y)
    # encoding = X,y
    X = data[0] + X
    # print(X)
    y = data[1] + y
    # print(y)
    # print(X)
    # print(y)
    data[0] = X
    data[1] = y
    # print(data)
    storeData(data, data_model_path)
    # Determine how many neighbors to use for weighting in the KNN classifier
    if n_neighbors is None:
        n_neighbors = int(round(math.sqrt(len(X))))
        if verbose:
            print("Chose n_neighbors automatically:", n_neighbors)

    # Create and train the KNN classifier
    knn_clf = neighbors.KNeighborsClassifier(n_neighbors=n_neighbors, algorithm=knn_algo, weights='distance')
    knn_clf.fit(X, y)

    # data.append(knn_clf)
    # Save the trained KNN classifier
    if model_save_path is not None:
        with open(model_save_path, 'wb') as f:
            # pickle.dump(knn_clf, f)
            pickle.dump(knn_clf, f)

    return knn_clf


train_id = input("Enter training ID: ")
train_dir = os.path.join(dataset_dir, train_id)
train(train_dir, train_id, model_save_path=save_model_path, n_neighbors=None, knn_algo='ball_tree', verbose=False)

# loadData(data_model_path)