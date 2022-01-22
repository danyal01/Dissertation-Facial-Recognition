#!/usr/bin/env python
# coding: utf-8

# # Facial Recognition

# In[31]:


import sys
sys.path.append('C:/Users/Danyal/AppData/Local/Programs/Python/Python39/Lib/site-packages')
import cv2
import os
import numpy as np
import face_recognition
import pygame
import datetime
import time 

# Declaring Variables 
path = 'KnownImages'
knownImages = []
studentNames = []
encodedList = []
myList = os.listdir(path)

pTime = 0
name = "Unkown"
currentFrameImg = face_recognition.load_image_file("UnknownImages/22_01_2022-03_22_10_PM.jpg")
currentLocImg = face_recognition.face_locations(currentFrameImg)
currentEncodeImg = face_recognition.face_encodings(currentFrameImg)[0]

# Importing all Images and appending filename to list
for i in myList:
    current = cv2.imread(f'{path}/{i}')
    knownImages.append(current)
    studentNames.append(os.path.splitext(i)[0])
print(studentNames)


# Encoding all the Known Images
def encodeImages(images):
    for image in knownImages:
        currentEncode = face_recognition.face_encodings(image)[0]
        encodedList.append(currentEncode)
    return encodedList

# Marking students attendance 
def attendanceMark(name):
    with open('Attendance.csv','r+') as attFile:
        AttendanceList = attFile.readlines()
        nameList = []
        for line in AttendanceList:
            pointer = line.split(',')
            nameList.append(pointer[0])
        if name not in nameList:
            currentTime = datetime.datetime.now().strftime('%H:%M:%S')
            attFile.writelines(f'\n{name},{currentTime}')
           
        
encodedKnownList = encodeImages(knownImages)
print('Encoding Complete')

# Calling the sound which indicates a match
pygame.mixer.init()
pygame.mixer.set_num_channels(8)
voice = pygame.mixer.Channel(2)
correctSound = pygame.mixer.Sound("Correct.mp3")

lastTime = datetime.datetime.now()
currentTime = datetime.datetime.now()

# Get user supplied values
cascPath = sys.argv[1]
# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
# Setting video source to the default webcam
video_capture = cv2.VideoCapture(0)

while True:
    # Capture frame-by-frame
    ret, frame = video_capture.read()  
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the video
    faces = (faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2, 
        minNeighbors=5,
        minSize=(50, 50),
        flags=cv2.CASCADE_SCALE_IMAGE)
    )
    
    """ For more accurate name tagging can use the current frame but FPS is too low """
    #currentFrameLoc = face_recognition.face_locations(frame)
    #currentFrameEnc = face_recognition.face_encodings(frame, currentFrameLoc)
    
    # Calculating FPS
    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime 
    cv2.putText(frame,f'FPS: {int(fps)}',(20, 50),cv2.FONT_HERSHEY_PLAIN,2,(255,255,255), 2)

    # Plays Sound, if there is a face and previous sound isnt playing  
    if len(faces) > 0:
        if voice.get_busy() == False:
            if(currentTime - lastTime).seconds > 1.5:
                lastTime = datetime.datetime.now()
                voice.play(correctSound)
              
            
    # Comparing face encodings, Lower Distance = face is more similar
    for i,y in zip(currentEncodeImg, currentLocImg):
        results = face_recognition.compare_faces(encodedKnownList, i)
        distance = face_recognition.face_distance(encodedKnownList, i)
        #print(distance)
        resultsIndex = np.argmin(distance)
        
        if results[resultsIndex]:
            name = studentNames[resultsIndex].upper()
            attendanceMark(name)
                
    # Draw a rectangle around the faces, Save and encode Frame 
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.rectangle(frame, (x-2, y+h+40), (x+w, y+h), (0, 255, 0), cv2.FILLED)
        cv2.putText(frame,name,(x+12, y+h+28),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255), 2)
        filename = datetime.datetime.now().strftime("%d_%m_%Y-%I_%M_%S_%p")
        cv2.imwrite('UnknownImages/' + str(filename) +'.jpg',frame) 
        currentFrameImg = face_recognition.load_image_file('UnknownImages/' + str(filename) +'.jpg')
        currentEncodeImg = face_recognition.face_encodings(currentFrameImg)
        currentTime = datetime.datetime.now()
        
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    # Exists if q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
        
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

#Need to boost FPS
#Need to change the classifier to improve speed
#Need to review the name saving to ensure student can get closest to time and mark as present
#Need to use dataset
#Need to apply ML algorithm


# In[ ]:




