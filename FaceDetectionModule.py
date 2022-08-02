'''
I will make module which called FaceDetectionModule.py to use able to next projects.
'''
import cv2 as cv
import mediapipe as mp
import time


# cap = cv.VideoCapture('I_like_this.mp4')
class FaceDetector():
    def __init__(self, min_detection_confidence=0.75, model_selection=1):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        

        self.mpFaceDetection = mp.solutions.face_detection
        self.mpDraw = mp.solutions.drawing_utils # The function is reached to draw
        self.faceDetection = self.mpFaceDetection.FaceDetection(self.min_detection_confidence,self.model_selection) # The fails are destroyed by 0.75


    
        
    
        
    def findFace(self,img,draw = True):
        imgRGB = cv.cvtColor(img,cv.COLOR_BGR2RGB) # The color is converted from BGR to RGB
        
        self.results = self.faceDetection.process(imgRGB) # The picture which is converted, is processed
    # print(results)
        boxs = []
        if self.results.detections: # The data which is face location, is kept as detections list.
            for id, detection in enumerate(self.results.detections): # Each face data is passed to detection variable
                
            
                #self.mpDraw.draw_detection(img,detection) # It is disappeared detection points
                # print(id,detection)
            # The datas which is in list, is written
                # print(detection.score)
                # print(detection.location_data.relative_bounding_box)
                bboxC = detection.location_data.relative_bounding_box # The object is created to know bounding state
                ih,iw,ic = img.shape
                # print(ih,iw,ic)
            
                
            # This method is made by youtuber
            # box1 = int(bboxC.xmin * iw),int(bboxC.ymin*ih),\
            #        int(bboxC.width*iw), int(bboxC.height*ih)
                
            # This method is made by me
                box2 = (int(bboxC.xmin*iw),int(bboxC.ymin*ih))
                box3 = (int(bboxC.xmin*iw)+int(bboxC.width*iw)),int(bboxC.ymin*ih)+int(bboxC.height*ih)
                boxs.append([id,box2,box3,detection.score])
                if draw:
                    
                    img = self.fancyDraw(img,box2,box3)
            
                    cv.putText(img,(f'{int(detection.score[0]*100)}%'),((box2[0]-10),(box2[1]-15)),cv.FONT_HERSHEY_COMPLEX,1,(250,50,50),2 )
                
        return img,boxs
            
            #imgg = cv.resize(img,(360,540)) # The picture which is got from file, is resized determined measurement
            # To find the fps
    def fancyDraw(self,img,boxLeftTop,boxRightDown,t = 10,rt =1):
        xLeft,yTop = boxLeftTop
        xRight,yDown = boxRightDown
        
        pointLT = (xLeft,yTop)
        pointLD = (xLeft,yDown)
        pointRT = (xRight,yTop)
        pointRD = (xRight,yDown)
        fancyColor =(255,0,255)
        
        cv.rectangle(img,boxLeftTop,boxRightDown,(250,0,250),rt)
        # to make corner lines
        # TOP LEFT
        cv.line(img,pointLT,((xLeft+30),yTop),fancyColor,t)
        cv.line(img,pointLT,(xLeft,(yTop+30)),fancyColor,t)
        # TOP RIGHT
        cv.line(img,pointRT,((xRight-30),yTop),fancyColor,t)
        cv.line(img,pointRT,(xRight,(yTop+30)),fancyColor,t)
        # DOWN LEFT
        cv.line(img,pointLD,((xLeft+30),yDown),fancyColor,t)
        cv.line(img,pointLD,(xLeft,(yDown-30)),fancyColor,t)
        # DOWN RIGHT
        cv.line(img,pointRD,((xRight-30),(yDown)),fancyColor,t)
        cv.line(img,pointRD,(xRight,(yDown-30)),fancyColor,t)
                
            
            
        return img

            
def main():
        cap = cv.VideoCapture('HASBULLA AND KHABIB.mp4')
        detector = FaceDetector()
        
        cTime = 0
        pTime = 0

        while True:
            success , img = cap.read() # The picture is read 
            
            cTime = time.time() 
            # print(cTime)
            # print(pTime)
            fps = 1/(cTime-pTime)
        
            pTime=cTime
        
            # print(fps)
            
            img = cv.putText(img,f'FPS: {int(fps)}',(20,40),cv.FONT_HERSHEY_PLAIN,2,(250,50,50),2)
            
            img,boxs = detector.findFace(img,True) # Second parameter decided whether lines are drawing
            print(boxs)
            cv.imshow('image',img)
    
            if cv.waitKey(20) & 0xFF == ord('a'):
                break
        

if __name__ == "__main__":
    main() 

    
else:
    print('This main function did not execute.')
    cv.destroyAllWindows()
    
