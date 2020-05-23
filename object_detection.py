import cv2, time, datetime
import numpy as np
from imutils.video import FPS

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