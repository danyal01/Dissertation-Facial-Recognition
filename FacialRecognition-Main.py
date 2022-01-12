#!/usr/bin/env python
# coding: utf-8

# In[15]:


import sys
sys.path.append('C:/Users/Danyal/AppData/Local/Programs/Python/Python39/Lib/site-packages')
import cv2
#print(cv2.__version__)
import datetime
import pygame


# # Facial Recognition

# In[22]:


# Setting successfull sound que
pygame.mixer.init()
pygame.mixer.set_num_channels(8)
voice = pygame.mixer.Channel(2)
correctSound = pygame.mixer.Sound("Correct.mp3")

# Recording the time to ensure the sound isnt played continuously
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
        scaleFactor=1.2, #scaleFactor=1.1
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    )
    
    # Plays Sound, if there is a face and previous sound isnt playing  
    if len(faces) > 0:
        if voice.get_busy() == False:
            if(currentTime - lastTime).seconds > 1.5:
                lastTime = datetime.datetime.now()
                voice.play(correctSound)
                
    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        filename = datetime.datetime.now().strftime("%d_%m_%Y-%I_%M_%S_%p")
        cv2.imwrite('StudentImages/ ' + str(filename) +'.jpg',frame) 
        currentTime = datetime.datetime.now()
        
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    #Exists if the q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
        
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()

