# Final tribrid to be run on Raspi.:)
import dlib
import face_recognition
import cv2, pickle
from firebase import firebase
import time, copy, datetime
from datetime import date
import numpy as np
from imutils.video import FPS

# initialize firebase url
firebase = firebase.FirebaseApplication('https://capstone-prototype-7b1f9.firebaseio.com/', None)
# serialized facial encodings
data = pickle.loads(open("encodings.pickle","rb").read())
detector = dlib.get_frontal_face_detector() # face detector
sdThresh = 8                                # thresh for standard deviation 
font = cv2.FONT_HERSHEY_SIMPLEX             # cv2 font general

# distMap returns pythogorean distance between two frames
def distMap(frame1, frame2):
    diff32 = np.float32(frame1) - np.float32(frame2)
    return np.uint8((np.sqrt(diff32[:,:,0]**2 + diff32[:,:,1]**2 + diff32[:,:,2]**2)/441.6729559300637)*255)

cap = cv2.VideoCapture(0)                   # general videocapture from default camera
_, frame1 = cap.read()                      # capturing first frame
_, frame2 = cap.read()                      # capturing second frame

time1 = time.time()
activity_count = 0
text = "Unoccupied"

fps = FPS().start()
# Main Loop
while(True):
    #TODO: Activity Monitoring
    _, frame = cap.read()                           # capture image
    rows, cols, _ = np.shape(frame)                 # get length & width of image
    dist = distMap(frame1, frame)                   # compute pythogorean distance
    frame1 = frame2                                 # reassign x[-2] frame
    frame2 = frame                                  # reassign x[-1] frame
    mod = cv2.GaussianBlur(dist, (9,9), 0)          # Apply gaussian smoothing
    _, thresh = cv2.threshold(mod, 100, 255, 0)     # Thresholding
    _, stDev = cv2.meanStdDev(mod)                  # calculate std deviation test

    if stDev > sdThresh: activity_count+=1          # computing activity intensity
    # push to motion detection data to cloud
    if(time.time()-time1>=5):
            time1 = time.time()
            nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            firebase.patch('/Motion Detection/',{nowtime:activity_count})
            activity_count=0
            print("activity detected")

    #TODO: Facial Recognition
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # grayscale image
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)    # rgb image

    dets = detector(rgb, 1)
    boxes = [(d.left(), d.top(), d.right(), d.bottom()) for i,d in enumerate(dets)]    # get tuple of box coordinates
    encodings = face_recognition.face_encodings(rgb, boxes)     # encode those faces from rgb
    
    names=[]
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
            if name=="P088": text = 'Occupied'

        # patching data to firebase console
        x=datetime.datetime.now().strftime("%H:%M:%S")
        y=datetime.datetime.now().strftime("%Y-%m-%d")
        firebase.patch('/Monitoring/'+name+'/'+y+'/',{x:"At camera 1"})

    #TODO: object detection
    # if len(gun) > 0:
    #     nowtime = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    #     firebase.patch("/Object Detection/",{nowtime:"gun detected at camera 1"})

    if cv2.waitKey(1) & 0xFF == 27: break       # break if esc is pressed

    #TODO: room occupacy status
    today=date.today().strftime("%m:%d:%Y")
    if text == 'Occupied':
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        firebase.patch('/Room Occupied/R0088/'+today,{current_time:text})
        result=firebase.get('/Room Occupied/R0088/'+today,'Occupied')
        if(result==None):
            firebase.patch('/Room Occupied/R0088/'+today,{'Occupied':1})
        else:
            result=firebase.get('/Room Occupied/R0088/'+today,'Occupied')
            result+=1
            firebase.patch('/Room Occupied/R0088/'+today,{'Occupied':result})
        print("Room is occupied")
    else:
        t = time.localtime()
        current_time = time.strftime("%H:%M:%S", t)
        firebase.patch('/Room Occupied/R0088/'+today,{current_time:text})
        result=firebase.get('/Room Occupied/R0088/'+today,'Unoccupied')
        if(result==None):
            firebase.patch('/Room Occupied/R0088/'+today,{'Unoccupied':1})
        else:
            result=firebase.get('/Room Occupied/R0088/'+today,'Unoccupied')
            result+=1
            firebase.patch('/Room Occupied/R0088/'+today,{'Unoccupied':result})
        print("Room is not occupied")

    fps.update()

# stop the timer and display FPS information
fps.stop()
print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

cap.release()
cv2.destroyAllWindows()