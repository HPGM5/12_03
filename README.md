import cv2
import numpy as np 
from matplotlib import pyplot as plt 

class compareImg : 
       def __init__(self) : 
       pass

       def readImg(self, filepath) : img = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
       
        return img

       def diffImg(self, img1, img2) :

           diffImg(self, img1, img2) :

            kp1, des1 = orb.detectAndCompute(img1, None)
            kp2, des2 = orb.detectAndCompute(img2, None)

            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            matches = bf.match(des1,des2)
            
            matches = sorted(matches, key = lambda x:x.distance)

            bf = cv2.BFMatcher() 
            matches = bf.knnMatch(des1, des2, k=2)
            good = [] 
              for m,n in matches: 
                  if m.distance < 0.75 * n.distance: 
                      good.append([m]) # Draw first 10 matches. knn_image = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good, None, flags=2)
                      plt.imshow(knn_image) 
                      plt.show()
