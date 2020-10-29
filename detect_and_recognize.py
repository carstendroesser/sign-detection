import os

import tensorflow as tf
from matplotlib import pyplot as plt

import plt_utils
from detector import detect_signs
from detector.detection_utils import *


def put_annotation(image, position, prediction):
    label = str.rstrip(labels[prediction[0]])[:20]
    probability = (str(prediction[1])[:4])
    annotation = label + ' @ ' + probability

    (text_width, text_height) = cv2.getTextSize(annotation, cv2.FONT_HERSHEY_PLAIN, fontScale=0.8, thickness=1)[0]
    # make the coords of the box with a small padding of two pixels
    box_coords = ((position[0], position[1]), (position[0] + text_width + 2, position[1] - text_height - 2))
    cv2.rectangle(image, box_coords[0], box_coords[1], (0, 255, 0), cv2.FILLED)
    cv2.putText(image, str.rstrip(annotation), (position[0], position[1]), cv2.FONT_HERSHEY_PLAIN, fontScale=0.8,
                color=(0, 0, 0), thickness=1)


plt_utils.setup(plt=plt, figsize=(5, 3), dpi=300)

# read all labels
file = open('sign_labels.txt', "r")
labels = []
for line in file:
    labels.append(str(line))
file.close()

# setup model
model = tf.keras.models.load_model('imported_models/sign_recognition')

samples = 'sample_frames'

for r, d, files in os.walk(samples):
    files = np.sort(files)
    for file in files:
        if '.jpg' in file:
            path_img = os.path.join(samples, file)
            image_orig = cv2.imread(filename=path_img)

            boxes = detect_signs(image_orig)
            # convert to RGB
            image_orig = cv2.cvtColor(src=image_orig, code=cv2.COLOR_BGR2RGB)

            predictions = []

            # cut out each box and predict it
            for box in boxes:
                # cut out
                roi = image_orig[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])]

                # prepare for model
                roi = cv2.cvtColor(roi, cv2.COLOR_RGB2GRAY)
                roi = crop_to_square_and_resize(roi, [32, 32])
                # although it is needed for training, it results in less inference-performance
                # roi = cv2.equalizeHist(roi)
                roi = tf.cast(roi, tf.float32) * (1.0 / 255.0)
                roi = tf.expand_dims(roi, -1)
                roi = tf.expand_dims(roi, 0)

                # predict
                prediction = model.predict(roi)
                sign_type = np.argmax(prediction)
                probability = prediction[0][sign_type]

                predictions.append([sign_type, probability])

            for prediction, box in zip(predictions, boxes):
                if prediction[1] > 0.95:
                    cv2.rectangle(img=image_orig, pt1=(box[0], box[1]), pt2=(box[0] + box[2], (box[1] + box[3])),
                                  color=(0, 255, 0), thickness=2)
                    put_annotation(image_orig, (box[0], box[1]), prediction)

            plt_utils.show(plt, image=image_orig, title='')
