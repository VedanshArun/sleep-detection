from scipy.spatial import distance
from imutils import face_utils
import dlib
import cv2
import numpy as np
import time
import wave
import os
import playsound
from multiprocess import Process , Value
from picamera2 import Picamera2
from csv import writer
import datetime
piCam = Picamera2()
piCam.preview_configuration.main.size=(1280, 720)
piCam.preview_configuration.align()
piCam.preview_configuration.main.format='RGB888'
piCam.configure('preview')
piCam.start()

start = time.time()
end = time.time()
start2 = time.time()
end2 = time.time()
initial_ignored_frames = 0
prev = 0 
pause = False 
pause2 = True
temp = False
open_eye_frames = 0 
time_of_alerts = []
counter = 0 
addToCSV = True
addToCSV2 = True
def writeToCSV(newRow):
	with open('/home/bss/node-app/routes/files/log.csv','a') as f_object:
		writer_object = writer(f_object)
		writer_object.writerow(newRow)
		f_object.close()

def play_alarm(runAlarmValue):
	while True:
		if runAlarmValue.value == 1:
			playsound.playsound('/home/bss/node-app/routes/files/alarm.mp3', True)


def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
eye_threshold = 0.25
startTimeNoted = False
startTimeNoted2= False
consecutive_frames = 5
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

cap=cv2.VideoCapture(0)
count=0

if __name__ == '__main__':
	runAlarm = Value('i', 0)
	process = Process(target = play_alarm , args = [runAlarm])
	process.start()
	while True:
		frame= piCam.capture_array()
		prev = time.time()
		gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
		initial_ignored_frames = initial_ignored_frames + 1
		if initial_ignored_frames > 30:
			subjects = detect(gray, 0)
			if len(subjects) == 0:
				counter = counter+1
				if counter >=10:
					if startTimeNoted == False:
						addToCSV = True
						start = time.time()
						startTimeNoted = True
					runAlarm.value = True
					
			else:
				counter = 0 
				runAlarm.value = False
				if addToCSV == True:
					startTimeNoted = False
					end = time.time()
					addRow = [datetime.datetime.now().isoformat(), round(end-start,1)]
					writeToCSV(addRow)
					addToCSV = False
				for subject in subjects:
					shape = predict(gray, subject)
					shape = face_utils.shape_to_np(shape)#converting to NumPy Array

					leftEye = shape[lStart:lEnd]
					rightEye = shape[rStart:rEnd]
				
					if len(leftEye) == 0:
						leftEye = [0,0,0,0,0]
					if len(rightEye) == 0:
						rightEye = [0,0,0,0,0]

					left_eye_asp_ratio = eye_aspect_ratio(leftEye)
					right_eye_asp_ratio = eye_aspect_ratio(rightEye)

					eye_asp_ratio = (left_eye_asp_ratio + right_eye_asp_ratio) / 2.0

					leftEyeHull = cv2.convexHull(leftEye)
					rightEyeHull = cv2.convexHull(rightEye)
					cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
					cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

					if eye_asp_ratio < eye_threshold:
						count = count + 1
						if count >= consecutive_frames:
							# cv2.putText(frame, "****************ALERT!****************", (10, 30),
							# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
							# cv2.putText(frame, "****************ALERT!****************", (10,325),
							# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
							if startTimeNoted2 == False:
								addToCSV2 = True
								start2 = time.time()
								startTimeNoted2 = True
							runAlarm.value = True
							time_of_alerts.append(time.time())

					else:
						runAlarm.value = False
						if addToCSV2 == True:
							startTimeNoted2 = False
							end2 = time.time()
							addRow = [datetime.datetime.now().isoformat(), round(end2-start2,1)]
							writeToCSV(addRow)
							addToCSV2 = False
						count = 0 


		cv2.imshow("Frame", frame)
		k = cv2.waitKey(1) & 0xFF
		if k == 27 or k==ord("q"):
			break
		
	cv2.destroyAllWindows()
	# print(time_of_alerts)