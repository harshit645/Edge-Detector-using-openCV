import cv2
import numpy as np

video=cv2.VideoCapture(0)

while True:

    check,frame=video.read()

    #laplacian method for edge detection
    gray_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    blur_image=cv2.GaussianBlur(gray_frame,(5,5),0)

    #64 represent the double data type
    laplacian=cv2.Laplacian(blur_image,cv2.CV_64F)


    #convert BGR to HSV
    #Here R deontes least significant area
    #similarly in RGB ,B denotes least significant area
    HSV_frame=cv2.cvtColor(frame,cv2.COLOR_BGR2HSV)

    #here we use the most appropriate values for lower and upper boundary
    lower_red=np.array([30,150,50])   #lower boundary of the threshold region
    upper_red=np.array([255,255,180]) #upper boundary of the threshold region

    mask=cv2.inRange(HSV_frame,lower_red,upper_red)

    #bitwise AND of frame and mask
    res=cv2.bitwise_and(frame,frame,mask=mask)

    #main part of canny method
    canny_image=cv2.Canny(frame,100,155)

    #to show original_image
    cv2.imshow("original_image",frame)
    #to show the laplacian_iamge
    cv2.imshow("laplacian_image",laplacian)
    #to show the canny_image
    cv2.imshow("canny_image",canny_image)
    #to show the result image
    cv2.imshow("bitwiseoforiginalandmaskimage",res)

    key=cv2.waitKey(1)
    if key==ord('q'):
        break


#to release the video
video.release()

#closing all open windows
cv2.destroyAllWindows()
