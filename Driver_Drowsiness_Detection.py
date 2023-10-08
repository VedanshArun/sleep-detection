from scipy.spatial import distance
from imutils import face_utils
import imutils
import dlib
import cv2
import winsound 
import numpy as np
import time

initial_ignored_frames = 0
prev = 0 
pause = False 
pause2 = True
temp = False
open_eye_frames = 0 

def eye_aspect_ratio(eye):
	A = distance.euclidean(eye[1], eye[5])
	B = distance.euclidean(eye[2], eye[4])
	C = distance.euclidean(eye[0], eye[3])
	ear = (A + B) / (2.0 * C)
	return ear
	
eye_threshold = 0.25
consecutive_frames = 15
detect = dlib.get_frontal_face_detector()
predict = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
cap=cv2.VideoCapture(0)


count=0
subject_null_count = 0
while True:
	ret, frame=cap.read()
	prev = time.time()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
	
	initial_ignored_frames = initial_ignored_frames + 1
	if initial_ignored_frames > 30:
		subjects = detect(gray, 0)
		if subject_null_count >=15:
			
			if pause == False:
				winsound.Beep(2000,2000)
			else:
				pause = False
				temp = False
			subject_null_count = 0

		elif len(subjects) == 0:
			subject_null_count = subject_null_count + 1
		
		else:
			pause = True
			temp = True
			subject_null_count = 0
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

				# leftEyeHull = cv2.convexHull(leftEye)
				# rightEyeHull = cv2.convexHull(rightEye)
				# cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
				# cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

				if eye_asp_ratio < eye_threshold:
					count = count + 1
					if count >= consecutive_frames:
						# cv2.putText(frame, "****************ALERT!****************", (10, 30),
						# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						# cv2.putText(frame, "****************ALERT!****************", (10,325),
						# 	cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
						if pause2 == False:
							winsound.Beep(2000, 2000)
						else:
							pause2 = False

						count = 0  

				else:
					open_eye_frames = open_eye_frames + 1
					if open_eye_frames >= 30:
						pause2 = False
					else : 
						pause2 = True
					count = 0 


	cv2.imshow("Frame", frame)
	k = cv2.waitKey(1) & 0xFF
	if k == 27 or k==ord("q"):
		break
		
cv2.destroyAllWindows()
