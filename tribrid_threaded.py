import time, copy, datetime, pickle, cv2, dlib, sys, threading
import face_recognition
from firebase import firebase
from datetime import date
import numpy as np
from imutils.video import FPS, WebcamVideoStream
import concurrent.futures

room = sys.argv[1]
# initialize firebase url
firebase = firebase.FirebaseApplication(
	'https://capstone-prototype-7b1f9.firebaseio.com/', None)
# serialized facial encodings
data = pickle.loads(open("encodings.pickle", "rb").read())
detector = dlib.get_frontal_face_detector()     # face detector
sdThresh = 8                                    # thresh for standard deviation
testingTime=20
# general videocapture from default camera
cap = WebcamVideoStream(src=0).start() # threaded capture

# -----------------------------------------------------------------------------------------------
# TODO: Activity Monitoring
# -----------------------------------------------------------------------------------------------
def activity(cap, room, firebase, testingTime):
    sdThresh = 8                    # thresh for standard deviation
    fps = FPS().start()
    time_first = time.time()
    activity_count = 0
    frame1 = cap.read()

    time_test = time.time()
    while(True):
        frame2 = cap.read()
        diff32 = np.float32(frame2) - np.float32(frame1)
        dist = np.uint8((np.sqrt(diff32[:, :, 0]**2 + diff32[:, :, 1]**2 + diff32[:, :, 2]**2)/441.6729559300637)*255)
        mod = cv2.GaussianBlur(dist, (9, 9), 0)         # Apply gaussian smoothing
        _, thresh = cv2.threshold(mod, 100, 255, 0)     # Thresholding
        _, stDev = cv2.meanStdDev(mod)                  # calculate std deviation test

        if stDev > sdThresh: activity_count += 1          # computing activity intensity
        # push to motion detection data to cloud
        t = time.time()
        if(t-time_first >= 5):
                time_first = t
                nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                firebase.patch('/Motion Detection/', {nowtime: activity_count})
                print(f"activity Intensity - {activity_count}")
                activity_count = 0
        fps.update()
        if(time.time()-time_test>testingTime): break
    fps.stop()
    print("[process-1] elasped time: {:.2f}".format(fps.elapsed()))
    print("[process-1] approx. FPS: {:.2f}".format(fps.fps()))

# -----------------------------------------------------------------------------------------------
# TODO: Facial Recognition
# -----------------------------------------------------------------------------------------------
def facial_recognition(cap, room, firebase, data, testingTime):
    detector = dlib.get_frontal_face_detector()     # face detector

    fps = FPS().start()
    time_test = time.time()
    while(True):
        frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # rgb image

        dets = detector(rgb, 1)
        boxes = [(d.left(), d.top(), d.right(), d.bottom())\
                    for i, d in enumerate(dets)]        # get tuple of box coordinates
        encodings = face_recognition.face_encodings(rgb, boxes) # encode those faces from rgb
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
                text = 'Occupied'

            # patching data to firebase console
            x = datetime.datetime.now().strftime("%H:%M:%S")
            y = datetime.datetime.now().strftime("%Y-%m-%d")
            names.append(name)
        for i in names:
            firebase.patch('/Monitoring/'+i+'/'+y+'/', {x: "At camera "+room})
        print(*names, 'are found in', room)

        # TODO: room occupacy status
        today=date.today().strftime("%m:%d:%Y")
        if text == 'Occupied':
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            firebase.patch(f'/Room Occupied/{room}/'+today,{current_time:text})
            result=firebase.get(f'/Room Occupied/{room}/'+today,'Occupied')
            if(result==None):
                firebase.patch(f'/Room Occupied/{room}/'+today,{'Occupied':1})
            else:
                result=firebase.get(f'/Room Occupied/{room}/'+today,'Occupied')
                result+=1
                firebase.patch(f'/Room Occupied/{room}/'+today,{'Occupied':result})
        else:
            t = time.localtime()
            current_time = time.strftime("%H:%M:%S", t)
            firebase.patch(f'/Room Occupied/{room}/'+today,{current_time:text})
            result=firebase.get(f'/Room Occupied/{room}/'+today,'Unoccupied')
            if(result==None):
                firebase.patch(f'/Room Occupied/{room}/'+today,{'Unoccupied':1})
            else:
                result=firebase.get(f'/Room Occupied/{room}/'+today,'Unoccupied')
                result+=1
                firebase.patch(f'/Room Occupied/{room}/'+today,{'Unoccupied':result})
        
        fps.update()
        if(time.time()-time_test>testingTime): break
    fps.stop()
    print("[process-2] elasped time: {:.2f}".format(fps.elapsed()))
    print("[process-2] approx. FPS: {:.2f}".format(fps.fps()))

# -----------------------------------------------------------------------------------------------
# TODO: object detection
# -----------------------------------------------------------------------------------------------
def object_detection(cap, room, firebase, testingTime):
    confidence_score = 0.5
    threshold = 0.3
    labelsPath = "yolo\obj.names"
    LABELS = open(labelsPath).read().strip().split("\n")
    weightsPath = "yolo\obj.weights"
    configPath = "yolo\obj.cfg"
    # Load YOLO and get output layer names
    net = cv2.dnn.readNetFromDarknet(configPath, weightsPath)
    ln = net.getLayerNames()
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]
    (W, H) = (None, None)

    fps = FPS().start()
    frame = cap.read()
    (H, W) = frame.shape[:2]

    time_test = time.time()
    while True:
        frame = cap.read()
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # rgb image
        blob = cv2.dnn.blobFromImage(
            rgb, 1 / 255.0, (416, 416), swapRB=True, crop=False)
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
                firebase.patch("/Object Detection/",{nowtime:f"{detected_object} detected at camera {room}"})
                print(f"{detected_object} found in room {room}")
        if cv2.waitKey(1) & 0xFF == 27: break       # break if esc is pressed
        fps.update()
        if(time.time()-time_test>testingTime): break
    fps.stop()
    print("[process-3] elasped time: {:.2f}".format(fps.elapsed()))
    print("[process-3] approx. FPS: {:.2f}".format(fps.fps()))


t1 = threading.Thread(target=activity, args=[cap, room, firebase, testingTime])
t2 = threading.Thread(target=facial_recognition, args=[cap, room, firebase, data, testingTime])
t3 = threading.Thread(target=object_detection, args=[cap, room, firebase, testingTime])
t1.start()
t2.start()
t3.start()
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     f1 = executor.submit(activity, [cap, room, firebase])
#     f2 = executor.submit(facial_recognition,[cap, room, firebase, data])
#     f3 = executor.submit(object_detection,[cap, room, firebase])
cap.stop()
