from imutils.video import FPS
import time, datetime, cv2
import numpy as np

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
        frame1 = frame2
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