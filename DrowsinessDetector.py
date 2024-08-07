from tkinter import *
import tkinter
from scipy.spatial import distance as dist
from imutils import face_utils
from imutils.video import VideoStream
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound as play


main = tkinter.Tk()
main.title("Driver Drowsiness Monitoring")
main.geometry("500x400")

def EAR(drivereye):
    point1 = dist.euclidean(drivereye[1], drivereye[5])
    point2 = dist.euclidean(drivereye[2], drivereye[4])
    # compute the euclidean distance between the horizontal
    distance = dist.euclidean(drivereye[0], drivereye[3])
    # compute the eye aspect ratio
    ear_aspect_ratio = (point1 + point2) / (2.0 * distance)
    return ear_aspect_ratio

def MOR(drivermouth):
    # compute the euclidean distances between the horizontal
    point   = dist.euclidean(drivermouth[0], drivermouth[6])
    # compute the euclidean distances between the vertical
    point1  = dist.euclidean(drivermouth[2], drivermouth[10])
    point2  = dist.euclidean(drivermouth[4], drivermouth[8])
    # taking average
    Ypoint   = (point1+point2)/2.0
    # compute mouth aspect ratio
    mouth_aspect_ratio = Ypoint/point
    return mouth_aspect_ratio

def NLR(drivernose,NOSE_AVERAGE):
    #NOSE_AVERAGE =1.2
    point = dist.euclidean(drivernose[3], drivernose[0])
    nose_np = point / NOSE_AVERAGE
    return nose_np/75
    
def startMonitoring():
    NOSE_AVERAGE = 1
    EYE_AR_THRESHOLD =0
    pathlabel.config(text="          Webcam Connected Successfully")
    webcamera = cv2.VideoCapture(0)
    #vs = VideoStream(src=args["webcam"]).start()
    ap = argparse.ArgumentParser()
    ap.add_argument("-w", "--webcam", type=int, default=0,help="index of webcam on system")
    args = vars(ap.parse_args())
    vs = VideoStream(src=args["webcam"]).start()
    svm_predictor_path = 'SVMclassifier.dat'
    svm_detector = dlib.get_frontal_face_detector()
    svm_predictor = dlib.shape_predictor(svm_predictor_path)
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (sNos,eNos) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]
    print("SetUp loading...")
    print("Please keep ur HEAD STRAIGHT toward's camera")
    time.sleep(5)
    print("Taking values please wait...")
    for i in range(75):
        frame = vs.read()
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = svm_detector(gray, 0)
        for rect in rects:
            point1 = 0.0
            ear_sum=0.0
            shape = svm_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            nose = shape[sNos:eNos]
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)
            ear = (leftEAR + rightEAR) / 2.0
            ear_sum+=ear
            EYE_AR_THRESHOLD=ear_sum
            point = dist.euclidean(nose[3], nose[0])
            point1 += point
            NOSE_AVERAGE = point1
		    #print("Avrg: ", NOSE_AVERAGE)
    NOSE_AVERAGE= NOSE_AVERAGE/75
    print("EYE_AVERAGE: ",EYE_AR_THRESHOLD)
    EYE_AR_CONSEC_FRAMES = 10
    EYE_AR_THRESH = 0.25
    NOSE_LENGTH_UP = 0.7
    NOSE_LENGTH_DOWN = 1.1
    MOU_AR_THRESH = 0.75
    COUNTER1 = 0
    COUNTER = 0
    yawnStatus = False
    yawns = 0
    #svm_detector = dlib.get_frontal_face_detector()
    #svm_predictor = dlib.shape_predictor(svm_predictor_path)
    #(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    #(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
    (mStart, mEnd) = face_utils.FACIAL_LANDMARKS_IDXS["mouth"]
    #(sNos,eNos) = face_utils.FACIAL_LANDMARKS_IDXS["nose"]

    while True:
        frame = vs.read()
        frame = imutils.resize(frame, width=640)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_yawn_status = yawnStatus
        rects = svm_detector(gray, 0)
        for rect in rects:
            shape = svm_predictor(gray, rect)
            shape = face_utils.shape_to_np(shape)
            leftEye = shape[lStart:lEnd]
            rightEye = shape[rStart:rEnd]
            mouth = shape[mStart:mEnd]
            nose = shape[sNos:eNos]
            leftEAR = EAR(leftEye)
            rightEAR = EAR(rightEye)
            mouEAR = MOR(mouth)
            noseNLR = NLR(nose,NOSE_AVERAGE)
            ear = (leftEAR + rightEAR) / 2.0
            leftEyeHull = cv2.convexHull(leftEye)
            rightEyeHull = cv2.convexHull(rightEye)
            mouthHull = cv2.convexHull(mouth)
            noseHull = cv2.convexHull(nose)
            cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 255), 1)
            cv2.drawContours(frame, [mouthHull], -1, (0, 255, 0), 1)
            cv2.drawContours(frame, [noseHull], -1, (0, 255, 0), 1)
            if noseNLR>NOSE_LENGTH_DOWN or noseNLR<NOSE_LENGTH_UP:
                #print("Head Bending")
                COUNTER1 = COUNTER1+1
                cv2.putText(frame, "NLR: {:.2f}".format(noseNLR), (480,90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if COUNTER1>EYE_AR_CONSEC_FRAMES:
                    play.playsound('MP3.wav')
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 130),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    cv2.putText(frame, "Head Bending!", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print('nlr',noseNLR)
                    print('nose average', NOSE_AVERAGE)
                    #cv2.putText(frame, "NLR: {:.2f}".format(noseNLR), (480,90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                else:  
                    cv2.putText(frame, "Head Bending!", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            else:
                COUNTER1 = 0
                cv2.putText(frame, "NLR: {:.2f}".format(noseNLR), (480,90),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if ear < EYE_AR_THRESH:
                COUNTER += 1
                cv2.putText(frame, "Eyes Closed ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    cv2.putText(frame, "DROWSINESS ALERT!", (10, 50),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    print('ear',ear)
                    print('eye ar threshold', EYE_AR_THRESH)
                    play.playsound('MP3.wav')
            else:
                COUNTER = 0
                #print("EYE_AVERAGE: ",EYE_AR_THRESHOLD)
                cv2.putText(frame, "Eyes Open ", (10, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, "EAR: {:.2f}".format(ear), (480, 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            if mouEAR > MOU_AR_THRESH:
                cv2.putText(frame, "Yawning, DROWSINESS ALERT! ", (10, 70),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print('mouth ear', mouEAR)
                print('mouth ar thresh', MOU_AR_THRESH)
                play.playsound('MP3.wav')
                yawnStatus = True
                output_text = "Yawn Count: " + str(yawns + 1)
                cv2.putText(frame, output_text, (10,100),cv2.FONT_HERSHEY_SIMPLEX, 0.7,(255,0,0),2)
            else:
                yawnStatus = False
            if prev_yawn_status == True and yawnStatus == False:
                yawns+=1
            cv2.putText(frame, "MOR: {:.2f}".format(mouEAR), (480, 60),cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            #cv2.putText(frame,"Visual Behaviour & Machine Learning Drowsiness Detection @ Drowsiness",(370,470),cv2.FONT_HERSHEY_COMPLEX,0.6,(153,51,102),1)
        cv2.imshow("Frame", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
    cv2.destroyAllWindows()
    vs.stop()
    webcamera.release()    


  

font = ('times', 16, 'bold')
title = Label(main, text='Driver Drowsiness Monitoring System using Visual\n               Behaviour and Machine Learning',anchor=W, justify=LEFT)
title.config(bg='black', fg='white')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)


font1 = ('times', 14, 'bold')
upload = Button(main, text="Start Behaviour Monitoring Using Webcam", command=startMonitoring)
upload.place(x=50,y=200)
upload.config(font=font1)  

pathlabel = Label(main)
pathlabel.config(bg='DarkOrange1', fg='white')  
pathlabel.config(font=font1)           
pathlabel.place(x=50,y=250)


main.config(bg='chocolate1')
main.mainloop()
