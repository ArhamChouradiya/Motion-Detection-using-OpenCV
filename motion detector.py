import cv2
import pandas
from datetime import datetime
import imutils

first_frame = None
status_list=[None,None]
times=[]

video = cv2.VideoCapture(0)
fourcc=cv2.VideoWriter_fourcc(*'XVID')
out=cv2.VideoWriter('Output.avi',fourcc,32,(640,480))
df=pandas.DataFrame(columns=['Start','End'])
while True:
    check,frame=video.read()
    status=0
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray,(5,5),0)
    if first_frame is None:
        first_frame = gray
        continue
    delta_frame = cv2.absdiff(first_frame,gray)
    thresh_delta = cv2.threshold(delta_frame,30,255,cv2.THRESH_BINARY)[1]
    thresh_delta = cv2.dilate(thresh_delta,None, iterations=2)
    cnts, hierarchy = cv2.findContours(thresh_delta.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    for contour in cnts:
        if cv2.contourArea(contour) < 700:
            continue
        status=1
        M = cv2.moments(contour)
        cX = int(M["m10"] / M["m00"])
        cY = int(M["m01"] / M["m00"])
        (x,y,w,h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x,y),(x+w,y+h),(0,255,0),3)
    
    status_list.append(status)
    status_list=status_list[-2:]
    if status_list[-1]==0 and status_list[-2]==1:
        times.append(datetime.now())
    if status_list[-1]==1 and status_list[-2]==0:
        times.append(datetime.now())
    if status_list[-1]==0 and status_list[-2]==0:
        cv2.putText(frame,"No Motion.",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    if status_list[-1]==1 and status_list[-2]==1:
        cv2.putText(frame,"Motion Detected!",(10,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,0,255),2,cv2.LINE_AA)
    cv2.imshow('frame',frame)
    cv2.imshow('Capturing',gray)
    cv2.imshow('delta',delta_frame)
    cv2.imshow('thresh',thresh_delta)
    out.write(frame)
    #first_frame = gray
    key =cv2.waitKey(10)
    if key == ord('q'):
        break

print(status_list)
print(times)

for i in range (0,len(times),2):
    df=df.append({"Start":times[i],"End":times[i+1]},ignore_index=True)

df.to_csv("Times.csv")

video.release()
out.release()
cv2.destroyAllWindows()