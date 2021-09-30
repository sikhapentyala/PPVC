from keras.utils import to_categorical
from keras.callbacks import EarlyStopping
from keras.callbacks import ModelCheckpoint
#from keras.optimizers import SGD
#from tensorflow.keras.optimizers import Adadelta
from tensorflow.keras.optimizers import SGD
#import SmallEmotCNN as vcModel
import gc

import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, AveragePooling2D, MaxPooling2D, MaxPool2D, Activation, Flatten, Dropout, \
    Dense
from tensorflow.keras.models import load_model
import sys
import logging
from tensorflow.keras.optimizers import SGD,Adam
import os
import sys
from os import listdir
from os.path import isfile, join

import cv2
from keras_preprocessing.image import img_to_array
from mtcnn import MTCNN
import GlobalVars as G


import numpy as np

import traceback
import logging

detector = MTCNN()

def build(num_classes=7):
    try:
        model = Sequential()

        model.add(Conv2D(64, (5, 5), padding="valid", input_shape=(48, 48, 1),strides=(1,1),activation="relu"))
        #model.add(Activation("relu"))
        model.add(AveragePooling2D(pool_size=(5, 5),strides=(2,2)))


        model.add(Conv2D(64, (3, 3), padding="valid",strides=(1,1),activation="relu"))
        #model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="valid",strides=(1,1),activation="relu"))
        #model.add(Activation("relu"))
        model.add(AveragePooling2D(pool_size=(3, 3),strides=(2,2)))

        model.add(Conv2D(128, (3, 3), padding="valid",strides=(1,1),activation="relu"))
        #model.add(Activation("relu"))
        model.add(Conv2D(128, (3, 3), padding="valid",strides=(1,1),activation="relu"))
        #model.add(Activation("relu"))
        model.add(AveragePooling2D(pool_size=(3, 3),strides=(2,2)))


        # Fully connected hidden layer
        model.add(Flatten())
        model.add(Dense(1024,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(1024,activation="relu"))
        model.add(Dropout(0.2))
        model.add(Dense(num_classes))
        model.add(Activation("softmax"))



        for srcLayer, dstLayer in zip(model_j.layers, model.layers):
            print("Before--->",srcLayer.name, dstLayer.name)
            if len(srcLayer.get_weights()) > 0:
                print(srcLayer.name,dstLayer.name)
                dstLayer.set_weights(srcLayer.get_weights())

        print(model.summary())


    except:
        logging.warning("Some unexpected error occurred Model not built fully:" + sys.exc_info()[1] + sys.exc_info()[0])
        exit(2)
    return model

def prepModel():
    with open('./models/model.json', 'r') as json_file:
        json_savedModel= json_file.read()
    #load the model architecture
    model_j = tf.keras.models.model_from_json(json_savedModel)
    model_j.summary()
    model_j.load_weights("./models/facial_expression_model_weights.h5")
    model_j.save('./models/pretrained.hd5')
    print(model_j.summary())
    t = build()
    t.compile(optimizer=SGD(),loss="categorical_crossentropy", metrics=["accuracy"])
    t.save("./models/changedModel.h5")




def pre_process_videos(files_to_prep, folder_name):
    global_faces = []
    global_labels = []

    try:
        for vid_name in files_to_prep:

            if not vid_name:
                logging.warning("video file name is empty")
                raise Exception("Video name not defined")

            vid_meta_data = vid_name.split("-")
            emotion = int(vid_meta_data[2])
            if emotion == 1:
                emotion_label = 6
            elif emotion == 3:
                emotion_label = 3
            elif emotion == 4:
                emotion_label = 4
            elif emotion == 5:
                emotion_label = 0
            elif emotion == 6:
                emotion_label = 2
            elif emotion == 7:
                emotion_label = 1
            elif emotion == 8:
                emotion_label = 5

            if folder_name ==  'train':
                PATH = G.TRAIN_PATH
            if folder_name == 'val':
                PATH = G.VAL_PATH

            vid_name_path = PATH + vid_name
            if not os.path.exists(vid_name_path):
                logging.warning("video file path not found: " + vid_name_path)
                raise Exception("Video file path not found")


            vid_reader = cv2.VideoCapture(vid_name_path)
            frame_num = 0
            while True:
                grabbed = vid_reader.grab()
                if grabbed:
                    if frame_num % 15 == 0:
                        vid_frame = vid_reader.retrieve()[1]
                        #vid_frame_gray = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)

                        faces = detector.detect_faces(vid_frame)
                        if faces is not None and len(faces) != 0:
                            if faces[0]['confidence'] > 0.98:

                                rows, cols = vid_frame.shape[:2]
                                rightEyeCenter = faces[0]['keypoints']['right_eye']
                                leftEyeCenter = faces[0]['keypoints']['left_eye']
                                dY = rightEyeCenter[1] - leftEyeCenter[1]
                                dX = rightEyeCenter[0] - leftEyeCenter[0]
                                angle = np.degrees(np.arctan2(dY, dX)) - 360
                                 # To align face rotate frame, and crop face and resize to 299,299
                                transform_Matrix = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
                                frame_rotated = cv2.warpAffine(vid_frame, transform_Matrix, (cols, rows))
                                if len(frame_rotated) == 0:
                                    logging.warning("Rotated Frame is empty")

                                faces_rot = detector.detect_faces(frame_rotated)
                                if faces_rot is not None and len(faces_rot) != 0:
                                    if faces_rot[0]['confidence'] > 0.98:
                                        (x, y, w, h) = faces_rot[0]['box']
                                        face_frame = cv2.resize(frame_rotated[y-0:y+h+0,
                                            x-0:x+w+0],
                                            (48,48), interpolation=cv2.INTER_CUBIC)
                                        face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)

                                        #cv2.imwrite(G.IMAGES_PATH  + vid_name[:-3] + "_"
                                        #    + str(frame_num) + ".jpg", face_frame)
                                        face_frame = img_to_array(face_frame)
                                        global_faces.append(face_frame)
                                        global_labels.append(emotion_label)

                        else:
                            logging.warning("No face detected in " + vid_name + "frame num: " + str(frame_num))
                else:
                    break
                frame_num += 1
            logging.warning("Preprocessed video: " + vid_name)
    except:
        logging.error("Exception raised in pre_process_video: " + str(sys.exc_info()[1]))
        traceback.print_exc()
    return global_faces, global_labels


def startTrain():
    try:
        if True:
            # Preprocess Training set
            files_to_prep_train = [f for f in listdir(G.TRAIN_PATH) if isfile(join(G.TRAIN_PATH, f)) and
                             (f[-4:] == '.mp4') ]
            train_images, train_labels = pre_process_videos(files_to_prep_train, "train")

            # Preprocess Val set
            files_to_prep_val = [f for f in listdir(G.VAL_PATH) if isfile(join(G.VAL_PATH, f)) and
                             (f[-4:] == '.mp4') ]
            val_images, val_labels = pre_process_videos(files_to_prep_val, "val")
        #else:
            #train_images, train_labels = prepProcess.read_image_database("train")
            #val_images, val_labels = prepProcess.read_image_database("val")

        #gc.collect()

        logging.warning("Preprocessing Completed. Start Normalize, one-hot encoding, ")

        # Normalize input
        train_images = np.array(train_images, dtype="float32") / 255.0
        val_images = np.array(val_images, dtype="float32") / 255.0

        train_labels = np.array(train_labels)
        val_labels = np.array(val_labels)

        # One-hot encoding
        #train_labels_categorical = to_categorical(train_labels, num_classes=len(G.EMOTION))
        #val_labels_categorical = to_categorical(val_labels, num_classes=len(G.EMOTION))
        train_labels_categorical = to_categorical(train_labels, num_classes=7)
        val_labels_categorical = to_categorical(val_labels, num_classes=7)

        # Standard variables use
        X_train = train_images
        X_test = val_images
        y_train = train_labels_categorical
        y_test = val_labels_categorical

        logging.warning("Start Build")
                # build model
        model = load_model("./models/changedModel.h5")
        if model is None:
            raise Exception("Model could not be built")

        # Use of augmentatation.
        # Use this for accuracy
        '''
        aug = ImageDataGenerator(rotation_range=25, width_shift_range=0.1,
                                 height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
                                 horizontal_flip=True, fill_mode="nearest")
        '''
        # Compile
        sgd = SGD(lr=G.lr, decay=G.DECAY, momentum=G.MOMENTUM, nesterov=True)
        #ada = Adadelta(learning_rate=1.0, rho=0.95)
        model.compile(loss="categorical_crossentropy", optimizer=sgd, metrics=["accuracy"])

        # Early Stopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=G.PATIENCE)
        mc = ModelCheckpoint(G.FINAL_MODEL, monitor='val_acc', mode='max', verbose=1, save_best_only=True)

        # Start Fit/ Training
        if (len(X_train) // G.BATCH) == 0:
            steps_per_epoch = 1
        else:
            steps_per_epoch = len(X_train) // G.BATCH
        logging.warning("Start Fit")
        training_history = model.fit(X_train, y_train, validation_data=(X_test, y_test),
                                   epochs=G.EPOCHS, batch_size=G.BATCH, verbose=1, callbacks=[es, mc])

        model.save("./models/savetrained.h5")
        # Plot training curves
        plot_model = training_history
        logging.warning("Plot")
        plt.style.use("ggplot")
        plt.figure()
        N = len(plot_model.history["loss"])

        plt.plot(np.arange(0, N), np.array(plot_model.history["loss"]), label="train_loss")
        plt.plot(np.arange(0, N), np.array(plot_model.history["val_loss"]), label="val_loss")
        plt.plot(np.arange(0, N), np.array(plot_model.history["accuracy"]), label="train_acc")
        plt.plot(np.arange(0, N), np.array(plot_model.history["val_accuracy"]), label="val_acc")

        plt.title("Training Loss and Accuracy")
        plt.xlabel("Epoch #")
        plt.ylabel("Loss/Accuracy")
        plt.legend(loc="upper right")

        # save plot to disk
        plt.savefig("PLOT"+".png")

    except:
        logging.error("Some unexpected error occurred Train:" + str(sys.exc_info()[1]))
        traceback.print_exc()
        sys.exit(2) 

prepModel()
startTrain()                                                                                                                                                                                                                                        