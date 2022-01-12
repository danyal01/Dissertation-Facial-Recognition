
# coding: utf-8

# In[1]:

#import vlc
import sys
sys.path.append('C:/Users/Danyal/AppData/Local/Programs/Python/Python39/Lib/site-packages')
import cv2
print(cv2.__version__)


# In[2]:

# Get user supplied values
cascPath = sys.argv[1]

# Create the haar cascade
faceCascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#p = vlc.MediaPlayer("Correct.mp3")

# Setting video source to the default webcam
video_capture = cv2.VideoCapture(0)

i=0

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

    # Draw a rectangle around the faces
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
        i+=1
        cv2.imwrite('StudentImages/Student' + str(i)+'.jpg',frame)
        #p.play()
        
    # Display the resulting frame
    cv2.imshow('Video', frame)
    
    
    #Exists if the q key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
        
# When everything is done, release the capture
video_capture.release()
cv2.destroyAllWindows()


# In[ ]:

#Play a sound for each sucessfull frame    pip3 install python-vlc
#Dont overwrite - unique number for each picture- date and time
#Timer - every 2 seconds save a picture 

