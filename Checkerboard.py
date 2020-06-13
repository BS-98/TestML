import numpy as np
import cv2

checkerboard = np.zeros(shape=(800,800,3), dtype=np.uint8)
WHITE  = (255, 255, 255)
NB_ROW = 8

for i in range(NB_ROW):
    row = i*100
    
    for column in range(100, 800, 200):
        
        # if row is even
        if i % 2 == 0: 
            checkerboard[row:row+100, column:column+100, :] = WHITE
            
        # if row is odd
        elif i % 2 != 0: 
            checkerboard[row:row+100, column-100:column, :] = WHITE

  
checkerboard[:, :, 2] = 0 #BGR
cv2.imshow("image", checkerboard)
cv2.waitKey()

