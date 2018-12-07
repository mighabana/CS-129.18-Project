import numpy as np
import cv2

#user input for the video number
print("Enter the Video to be analyzed (1,2, or 3):")
video_num = input()
while int(video_num) not in [1,2,3]:
	print("Not a valid number. Please enter either 1, 2, or 3:")
	video_num = input()

#user input for the car cascades
print("Which car cascade should we use?")
print("[1] The cascade we trained;\n[2] Online Cascade")
cascade_num = input()
while int(cascade_num) not in [1,2]:
	print("Not a  valid number. Please enter either 1 or 2")
	cascade_num = input()
if cascade_num == '1':
	cascade_num = 'myhaar13.xml'
else:
	cascade_num = 'car_side.xml'

print("Video and analysis is running. Press q to quit...")

cap = cv2.VideoCapture('video'+video_num+'.mp4')

#this is for identifying the cars
car_cascade = cv2.CascadeClassifier(cascade_num)
pedestrian_cascade = cv2.CascadeClassifier('pedestrian.xml')

ret,frame=cap.read()

out = cv2.VideoWriter('viola-jones_output.mp4',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (640,360))

#loop over all frames i.e. play the video
while(cap.isOpened()):
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #detect cars in video
    cars = car_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(90,90))

    #detect pedestrians in video
    pedestrians = pedestrian_cascade.detectMultiScale(gray, maxSize=(80,80), scaleFactor=1.05, minNeighbors=5) 

    #to draw green arectangle around each cars 
    for (x,y,w,h) in cars:
    	if y > 300 or video_num =='3':
        	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #print(x,y,w,h) 

    #to draw red rectangles around each pedestrian
    for (x,y,w,h) in pedestrians:
    	if (x < 200 and y>250) or video_num =='3' :
        	cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0, 255),2)
        #print(x,y,w,h)  

    cv2.imshow('Frame',frame)   
    out.write(frame)
    # cv2.waitKey(0)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


cap.release()
out.release()

cv2.destroyAllWindows()
