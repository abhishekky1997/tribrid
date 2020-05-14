# Final tribrid to be run on Raspi.:)
import dlib
import face_recognition
import cv2
import pickle
from firebase import firebase
import time
import copy
import datetime
from datetime import date
import numpy as np
from imutils.video import FPS
import sys

room = sys.argv[1]
# initialize firebase url
firebase = firebase.FirebaseApplication(
    'https://capstone-prototype-7b1f9.firebaseio.com/', None)
# serialized facial encodings
data = pickle.loads(open("encodings.pickle", "rb").read())
detector = dlib.get_frontal_face_detector()  # face detector
sdThresh = 8                                # thresh for standard deviation
font = cv2.FONT_HERSHEY_SIMPLEX             # cv2 font general

# -----------------------------------------------------------------------------------------------
# Object detection parameters
# -----------------------------------------------------------------------------------------------
confidence_score = 0.5
threshold = 0.3
labelsPath = "yolo/obj.names"
LABELS = open(labelsPath).read().strip().split("\n")
weightsPath = "yolo/obj.weights"
configPath = "yolo/obj.cfg"
# Load YOLO and get output layer names
net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
ln = net.getLayerNames()
ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
(W, H) = (None, None)

# -----------------------------------------------------------------------------------------------
# DistMap function -> return pythogorean distance between two frames
# -----------------------------------------------------------------------------------------------


def distMap(frame1, frame2):
    diff32 = np.float32(frame1) - np.float32(frame2)
    return np.uint8((np.sqrt(diff32[:, :, 0]**2 + diff32[:, :, 1]**2 + diff32[:, :, 2]**2)/441.6729559300637)*255)


# general videocapture from default camera
cap = cv2.VideoCapture(0)
grabbed, frame1 = cap.read()                # capturing first frame
# exit if unable to grab frames
if not grabbed:
    sys.exit("unable to grab frames, error in camera")
#  grab frame dimensions if they are empty
if W is None or H is None:
    (H, W) = frame1.shape[:2]
_, frame2 = cap.read()                      # capturing second frame

time1 = time.time()
activity_count = 0
# -----------------------------------------------------------------------------------------------

# TODO: Main Loop
fps = FPS().start()
while(True):
    # -----------------------------------------------------------------------------------------------
    # TODO: Activity Monitoring
    # -----------------------------------------------------------------------------------------------
    _, frame = cap.read()                           # capture image
    # get length & width of image
    rows, cols, _ = np.shape(frame)
    # compute pythogorean distance
    dist = distMap(frame1, frame)
    frame1 = frame2                                 # reassign x[-2] frame
    frame2 = frame                                  # reassign x[-1] frame
    mod = cv2.GaussianBlur(dist, (9, 9), 0)          # Apply gaussian smoothing
    _, thresh = cv2.threshold(mod, 100, 255, 0)     # Thresholding
    # calculate std deviation test
    _, stDev = cv2.meanStdDev(mod)

    if stDev > sdThresh:
        activity_count += 1          # computing activity intensity
    # push to motion detection data to cloud
    if(time.time()-time1 >= 5):
        time1 = time.time()
        nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        firebase.patch('/Motion Detection/', {nowtime: activity_count})
        activity_count = 0
        print("activity detected")

# -----------------------------------------------------------------------------------------------
# TODO: Facial Recognition
# -----------------------------------------------------------------------------------------------
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # rgb image

    dets = detector(rgb, 1)
    boxes = [(d.left(), d.top(), d.right(), d.bottom())
             for i, d in enumerate(dets)]        # get tuple of box coordinates
    encodings = face_recognition.face_encodings(
        rgb, boxes)  # encode those faces from rgb

    names = []
    text = "Unoccupied"
    # Loop over facial embeddings and check if faces match
    for encoding in encodings:
        matches = face_recognition.compare_faces(data["encodings"], encoding)
        name = "Unknown"
        if True in matches:
            matchedIdxs = [i for (i, b) in enumerate(matches) if b]
            counts = {}

            for i in matchedIdxs:
                name = data["names"][i]
                counts[name] = counts.get(name, 0) + 1
            name = max(counts, key=counts.get)
            print(f"{name} was detected in room R088")
            text = 'Occupied'

        # patching data to firebase console
        x = datetime.datetime.now().strftime("%H:%M:%S")
        y = datetime.datetime.now().strftime("%Y-%m-%d")
        firebase.patch('/Monitoring/'+name+'/'+y+'/', {x: "At camera "+room})
        names.append(name)
    print(*names, 'are found in ', room)

# -----------------------------------------------------------------------------------------------
# TODO: object detection
# -----------------------------------------------------------------------------------------------
    blob = cv2.dnn.blobFromImage(
        frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    layerOutputs = net.forward(ln)

    # initialize lists of detected bounding boxes, confidences, and class IDs, respectively
    boxes = []
    confidences = []
    classIDs = []

    # loop over each of the layer outputs
    for output in layerOutputs:
        # loop over each of the detections
        for detection in output:
            # extract class ID and confidence by using score for each class
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]

            # filter out weak predictions by ensuring the detected
            # probability is greater than the minimum probability
            if confidence > confidence_score:
                # YOLO returns center coords, width and height
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                # derive top left corner for NMS
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                # Update all
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

        # apply NMS
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, confidence_score, threshold)
    if len(idxs) > 0:  # if detected
        # loop over detected objects and push to cloud
        for i in idxs.flatten():
            detected_object = LABELS[classIDs[i]]
            nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            firebase.patch(
                "/Object Detection/", {nowtime: f"{detected_object} detected at camera {room}"})

# -----------------------------------------------------------------------------------------------
# TODO: room occupacy status
# -----------------------------------------------------------------------------------------------
    today = date.today().strftime("%m:%d:%Y")
    if text == 'Occupied':
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        firebase.patch(f'/Room Occupied/{room}/'+today, {current_time: text})
        result = firebase.get(f'/Room Occupied/{room}/'+today, 'Occupied')
        if(result == None):
            firebase.patch(f'/Room Occupied/{room}/'+today, {'Occupied': 1})
        else:
            result = firebase.get(f'/Room Occupied/{room}/'+today, 'Occupied')
            result += 1
            firebase.patch(
                f'/Room Occupied/{room}/'+today, {'Occupied': result})
    else:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        firebase.patch(f'/Room Occupied/{room}/'+today, {current_time: text})
        result = firebase.get(f'/Room Occupied/{room}/'+today, 'Unoccupied')
        if(result == None):
            firebase.patch(f'/Room Occupied/{room}/'+today, {'Unoccupied': 1})
        else:
            result = firebase.get(
                f'/Room Occupied/{room}/'+today, 'Unoccupied')
            result += 1
            firebase.patch(
                f'/Room Occupied/{room}/'+today, {'Unoccupied': result})

    if cv2.waitKey(1) & 0xFF == 27:
        break       # break if esc is pressed
    fps.update()
# -----------------------------------------------------------------------------------------------
# TODO: End loop
# -----------------------------------------------------------------------------------------------

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()
