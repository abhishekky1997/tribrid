import pickle, sys, threading
from firebase import firebase
from imutils.video import WebcamVideoStream

import object_detection
import facial_recognition
import activity

room = sys.argv[1]
# initialize firebase url
firebase = firebase.FirebaseApplication('https://capstone-prototype-7b1f9.firebaseio.com/', None)
# serialized facial encodings
data = pickle.loads(open("encodings.pickle", "rb").read())
# testing time of code
testingTime=20
# general videocapture from default camera
cap = WebcamVideoStream(src=0).start() # threaded capture

t1 = threading.Thread(target=activity, args=[cap, room, firebase, testingTime])
t2 = threading.Thread(target=facial_recognition, args=[cap, room, firebase, data, testingTime])
t3 = threading.Thread(target=object_detection, args=[cap, room, firebase, testingTime])
t1.start()
t2.start()
t3.start()
t1.join()
t2.join()
t3.join()
# with concurrent.futures.ProcessPoolExecutor() as executor:
#     f1 = executor.submit(activity, [cap, room, firebase])
#     f2 = executor.submit(facial_recognition,[cap, room, firebase, data])
#     f3 = executor.submit(object_detection,[cap, room, firebase])
cap.stop()
