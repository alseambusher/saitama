import sys
import cv2
sys.path.append("/usr/local/python")
from openpose import pyopenpose as op
import numpy as np
import time

cap = cv2.VideoCapture(0)
# ffmpeg -i VID_20210720_180616.mp4  -filter:v fps=fps=5 VID_20210720_180616_5fps.mp4
# cap = cv2.VideoCapture('VID_20210720_180616_5fps.mp4')
ret, frame1 = cap.read()
prvs = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
params = dict()
params["model_folder"] = "openpose/models/"
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()
colors = (np.reshape(np.random.random(size=3*25), (25, 3)) * 255).tolist()
height, width, _ = frame1.shape
fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v')
out = cv2.VideoWriter("out.mp4", fourcc, 5, (width, height))
old = []
while True:
    # next = cv2.cvtColor(frame2, cv2.COLOR_BGR2RGB)

    # Process Image
    datum = op.Datum()
    ret, frame = cap.read()
    datum.cvInputData = frame
    # imageToProcess = cv2.imread(args[0].image_path)
    # datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))

    # Display Image
    print("Body keypoints: \n" + str(datum.poseKeypoints))
    try:
        image = np.zeros((height, width, 3), np.uint8)
        points = datum.poseKeypoints[0]
        centers = []
        for idx, point in enumerate(points):
            center = (int(point[0]), int(point[1]))
            centers.append(center)
            image = cv2.circle(image, center, radius=10, color=colors[idx], thickness=-1)
        old.append(centers)
        for old_centers_idx, old_centers in enumerate(old):
            for idx, center in enumerate(old_centers):
                if old_centers_idx > 0:
                    image = cv2.line(image, old[old_centers_idx-1][idx], center, color=colors[idx], thickness=2)

        cv2.imshow("pose", image)
        print("Writing frame")
        out.write(image)
        print("Wrote frame")
        # cv2.imshow("OpenPose 1.7.0 - Tutorial Python API", datum.cvOutputData)
    except:
        print("No found")
    # frame = cv2.flip(frame, 0)

    # write the flipped frame
    k = cv2.waitKey(1) & 0xff
    if k == 27:
        break





