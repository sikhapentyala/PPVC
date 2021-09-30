import pathlib
import GlobalVars as G
import sys
import seaborn as sns
import logging
import traceback
import matplotlib.pyplot as plt
import cv2
from mtcnn import MTCNN
import numpy as np
from tensorflow.keras.models import load_model
from keras_preprocessing.image import img_to_array
from collections import Counter
import sklearn.metrics as sm
import os
import sys
from os import listdir
from os.path import isfile, join
import tensorflow as tf
import keract
import keras.backend as K

EMOTION = {'1': 'neutral', '2': 'calm', '3': 'happy', '4': 'sad', '5': 'angry',
           '6': 'fearful', '7': 'disgust', '8': 'surprised'}

def relu(x):
  if x > 0: return x
  else: return 0

def approx_softmax(emotion_inference_softmax):
  #print(emotion_inference_softmax.shape)
  approx_softmax=[]
  sum_logit = 0
  for logit in emotion_inference_softmax:
    sum_logit += relu(logit)
  if sum_logit > 0:
      for logit in emotion_inference_softmax:
          approx_softmax.append(relu(logit)/relu(sum_logit))
  else:
      return np.array([1/7]*emotion_inference_softmax.shape[0])
      #approx_softmax.fill(emotion_inference_softmax.shape[0])
  return np.array(approx_softmax)

   

def preprocess_video(path_to_video):
    faces_to_infer = []
    detector = MTCNN()
    try:
        vid_reader = cv2.VideoCapture(path_to_video)
        frame_num = 0
        num_frames_read = 0
        while True:
            grabbed = vid_reader.grab()
            if grabbed:
                if frame_num % 15 == 0:
                    vid_frame = vid_reader.retrieve()[1]
                    vid_frame_gray = cv2.cvtColor(vid_frame, cv2.COLOR_BGR2GRAY)

                    faces = detector.detect_faces(vid_frame)
                    if faces is not None and len(faces) != 0:
                        if faces[0]['confidence'] > 0.98:
                            rows, cols = vid_frame_gray.shape[:2]
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
                                    face_frame = cv2.resize(frame_rotated[y - 0:y + h + 0,
                                                            x - 0:x + w + 0],
                                                            (48, 48),
                                                            # (10,10),
                                                            interpolation=cv2.INTER_CUBIC)
                                    face_frame = cv2.cvtColor(face_frame, cv2.COLOR_BGR2GRAY)
                                    #cv2.imwrite("imagedatabase/" + "/PROC_"
                                    #            + str(frame_num) + ".jpg", face_frame)
                                    face_frame = img_to_array(face_frame)
                                    faces_to_infer.append(face_frame)
                                    num_frames_read += 1


                    else:
                        logging.warning("No face detected in " + "frame num: " + str(frame_num) + str(path_to_video))
            else:
                break
            frame_num += 1

        logging.warning("Preprocessed video")
    except:
        logging.error("Exception raised in pre_process_video: " + str(sys.exc_info()[1]))
        traceback.print_exc()
    return faces_to_infer, num_frames_read



def classifyVideoAvg(path_to_video):
    inference_max = ""
    inference_mean = ""
    try:

        #if not (pathlib.Path("ForQuant/models/trainedACT.h5").exists()):
        if not (pathlib.Path(G.FINAL_MODEL).exists()):
            logging.warning("No model found. Cannot infer")
            raise Exception("No model found. Cannot infer")

        #emotion_detection = load_model("ForQuant/models/trainedACT.h5")
        emotion_detection = load_model(G.FINAL_MODEL)

        frame_inferences = []
        frame_inferences_probability = []

        faces_to_infer, no_of_frames_infer = preprocess_video(path_to_video)
        if len(faces_to_infer) == 0:
            raise Exception("No faces detected in entire video")
        else:
            for face in faces_to_infer:
                face = np.array(face, dtype="float32") / 255.0
                face = np.expand_dims(face, axis=0)
 
                    
                emotion_inference_softmax = emotion_detection.predict(face)[0]
                emotion_inference_logits = keract.get_activations(emotion_detection, face,
                                     layer_names='dense_5')['dense_5'][0]
                emotion_inference_approx_softmax = approx_softmax(emotion_inference_logits)
                #print("Emot Inference Logits:", emotion_inference_logits)
                #print("Emot Inference for frame with Softmax: ", emotion_inference_softmax )
                #print("Emot Inference for frame Softmax approx ", emotion_inference_approx_softmax )
                frame_inferences_probability.append(emotion_inference_approx_softmax)

        # Approach using Rolling Average
        frame_inferences_np = np.array(frame_inferences_probability)
        #print("Frame probs:",frame_inferences_np)
        sum_emots = frame_inferences_np.sum(axis=0)
        #print("Sum emots:",sum_emots)       
        emotion_probability_max = np.max(sum_emots)
        emotion_label_arg = np.argmax(sum_emots)
        inference_mean = emotion_label_arg

        #inference_arg = frame_inferences[np.argmax(frame_inferences_probability)]

    except:
        logging.error("Some unexpected error occurred Make Inference:" + str(sys.exc_info()[1]))
        sys.exit(2)
    return emotion_label_arg, inference_mean



EMOTION_CLASS = {'6' : 'neutral', '3':'happy', '4':'sad', '0' :'angry',
           '2' : 'fearful', '1' : 'disgust', '5' : 'surprised'}
EMOTION = {'1' : 'neutral', '2' : 'calm' , '3':'happy', '4':'sad', '5' :'angry',
           '6' : 'fearful', '7' : 'disgust', '8' : 'surprised'}

#0       1       2     3      4    5         6
#angry, disgust, fear, happy, sad, surprise, neutral
#
#FOLDER = "ForQuant/TEST_ACT7/"

FOLDER = G.TEST_SET
files_to_read = [f for f in listdir(FOLDER) if isfile(join(FOLDER, f)) and
                 (f[-4:] == '.mp4')]
#files_to_read = ["02-01-07-02-01-01-06.mp4"]
tot = len(files_to_read)
print(tot)
corr = 0
y_truth = []
y_predict = []
for file in files_to_read:
    vid_meta_data = file.split("-")
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
    label, mn = classifyVideoAvg(FOLDER + file)
    y_truth.append(emotion_label)
    y_predict.append(label)
    if label == emotion_label:
        corr += 1
        print("Correct===============>",file,"==>", corr/tot)

print("acc With Approx Softmax on Test =========>", corr/tot)
conf = sm.confusion_matrix(y_truth, y_predict, labels=[0,1,2,3,4,5,6])
ax = plt.subplot()
sns.heatmap(conf, annot=True, ax=ax);  # annot=True to annotate cells
# labels, title and ticks
ax.set_xlabel('Predicted labels');
ax.set_ylabel('True labels');
ax.set_title('Confusion Matrix');
# plt.show()
plt.savefig("cf.png")
