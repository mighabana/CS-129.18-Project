import numpy as np
import cv2

vid = 0
fileName = ""
while not (vid == "1" or vid == "2" or vid == "3"): 
	print("Please select which video to run on: ")
	print("1 - Video1 || 2 - Video2")
	vid = input()
	if vid == "1":
		fileName = "video1.mp4"
		break
	elif vid == "2":
		fileName = "video2.mp4"
		break
	elif vid == "3":
		fileName = "video3.mp4"
		break

cap = cv2.VideoCapture(fileName)
ret, frame =  cap.read()

#We need the kernel for Binary Image Processing (Erosion, Dilation, Opening, Closing)
#kernel = np.ones((3,3), np.uint8)


#The image displays in landscape so we have to rotate it. 
#This is done by getting the rotation matrix and applying it to all the frames.
rows, cols, channels = frame.shape
rotM = cv2.getRotationMatrix2D((cols/2,rows/2),-90,0.5)

#choose the background subtractor algorithm(I just got this from the OpenCV docs -
#https://docs.opencv.org/3.4/d1/dc5/tutorial_background_subtraction.html)
bgSub = cv2.createBackgroundSubtractorMOG2()
# bgSub = cv2.createBackgroundSubtractorKNN()

#opening. This consists of erosion then dilation
def imOpen (binImg, kernelType = 'rect', kernelSize = 2, numIter = 1):
	if kernelType == 'ellipse':
		kernelT = cv2.MORPH_ELLIPSE
	elif kernelType == 'rect':
		kernelT = cv2.MORPH_RECT
	elif kernelType == 'cross':
		kernelT = cv2.MORPH_CROSS
	kernel = cv2.getStructuringElement(kernelT, (kernelSize,kernelSize))
	for i in range(numIter):
		binImg = cv2.morphologyEx(binImg, cv2.MORPH_OPEN, kernel)
	return binImg

def imClose (binImg, kernelType = 'rect', kernelSize = 2, numIter = 1):
	if kernelType == 'ellipse':
		kernelT = cv2.MORPH_ELLIPSE
	elif kernelType == 'rect':
		kernelT = cv2.MORPH_RECT
	elif kernelType == 'cross':
		kernelT = cv2.MORPH_CROSS
	kernel = cv2.getStructuringElement(kernelT, (kernelSize,kernelSize))
	for i in range(numIter):
		binImg = cv2.morphologyEx(binImg, cv2.MORPH_CLOSE, kernel)
	return binImg

def imfill (binImg):
	imgCopy = binImg
	h, w = binImg.shape[:2]
	mask = np.zeros((h+2,w+2), np.uint8)
	cv2.floodFill(imgCopy, mask, (0,0), 255)
	imgInvert = cv2.bitwise_not(binImg)
	return cv2.bitwise_not (binImg | imgInvert)

def drawBoxes (cars, pedestrians):
	for (x,y,w,h) in cars:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        #print(x,y,w,h) 

    #to draw red rectangles around each pedestrian
	for (x,y,w,h) in pedestrians:
		cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0, 255),2)
        #print(x,y,w,h)  

def BinaryThresholding(gray_img):
	ret, thresh = cv2.threshold(gray_img, 200, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
	th2 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)
	th3 = cv2.adaptiveThreshold(gray_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
	return thresh

while(cap.isOpened()):
	ret, frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	if(vid != "3"):
		frame = frame[200:]

	#get the forground from the selected bgsubtraction algorithm.
	#The foreground is binary
	foreground = bgSub.apply(frame, learningRate=0)
	foreground[foreground == 127] = 0 #change shadows to one (this will help in closing)

	# thresh = BinaryThresholding(foreground)
	opening = imOpen(foreground, kernelType='ellipse', kernelSize = 3, numIter = 3)
	closing = imClose(opening ,kernelType = 'ellipse', kernelSize = 3, numIter = 20)
	dilate = cv2.dilate(closing, None, iterations=3)
	cnts = cv2.findContours(dilate.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
	cnts = cnts[1]

	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) < 2000:
			continue
 
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

	for c in cnts:
		# if the contour is too small, ignore it
		if cv2.contourArea(c) > 600 or cv2.contourArea(c) < 350:
			continue
 
		# compute the bounding box for the contour, draw it on the frame,
		# and update the text
		(x, y, w, h) = cv2.boundingRect(c)
		cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)


	# Display whatever frames we have
	cv2.imshow('foreground', foreground)
	cv2.imshow('opening', opening)
	cv2.imshow('closing', closing)
	cv2.imshow('dilate', dilate)
	# cv2.imshow('opening_again', opening_again) 

	cv2.imshow('frame', frame)



	if cv2.waitKey(1) & 0xFF == ord('q'):
		break