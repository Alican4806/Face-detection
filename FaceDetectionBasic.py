

import cv2 as cv
import mediapipe as mp
import time


cap = cv.VideoCapture('I_like_this.mp4')


mpFaceDetection = mp.solutions.face_detection
mpDraw = mp.solutions.drawing_utils # The function is reached to draw
faceDetection = mpFaceDetection.FaceDetection(0.75,1) # The fails are destroyed by 0.75


cTime = 0
pTime = 0
while True:
    success, imgg = cap.read() # The picture is read 
    imgRGB = cv.cvtColor(imgg,cv.COLOR_BGR2RGB) # The color is converted from BGR to RGB
    
    results = faceDetection.process(imgRGB) # The picture which is converted, is processed
    # print(results)
    if results.detections: # The data which is face location, is kept as detections list.
        for id, detection in enumerate(results.detections): # Each face data is passed to detection variable
            
            
            #mpDraw.draw_detection(imgg,detection) # If this line is became comment, it will disappear detection points on the face.
            print(id,detection)
            # The datas which is in list, is written
            print(detection.score)
            print(detection.location_data.relative_bounding_box)
            bboxC = detection.location_data.relative_bounding_box # The object is created to know bounding state
            ih,iw,ic = imgg.shape
            print(ih,iw,ic)
            # This method is made by youtuber
            # box1 = int(bboxC.xmin * iw),int(bboxC.ymin*ih),\
            #        int(bboxC.width*iw), int(bboxC.height*ih)
                   
            # This method is made by me
            box2 = (int(bboxC.xmin*iw),int(bboxC.ymin*ih))
            box3 = (int(bboxC.xmin*iw)+int(bboxC.width*iw)),int(bboxC.ymin*ih)+int(bboxC.height*ih)
             
            
            cv.rectangle(imgg,box2,box3,(250,50,250),6)
            cv.putText(imgg,(f'{int(detection.score[0]*100)}%'),(box2[0],box3[1]-170),cv.FONT_HERSHEY_COMPLEX,1,(250,50,50),2 )
            
            
            
            #imgg = cv.resize(img,(360,540)) # The picture which is got from file, is resized determined measurement
            # To find the fps
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime=cTime
        print(fps)
            
    
        cv.putText(imgg,f'FPS: {int(fps)}',(20,40),cv.FONT_HERSHEY_PLAIN,2,(250,50,50),2)
    
        cv.imshow('image',imgg)
    
    
    if cv.waitKey(20) & 0xFF == ord('a'):
        break
    
cv.destroyAllWindows()
    