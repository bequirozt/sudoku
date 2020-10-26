import cv2 as cv
import matplotlib.pyplot as plt

img = cv.imread('img/sudoku.jpeg')

color = cv.cvtColor(img,cv.COLOR_BGR2HSV)[:,:,2]

plt.imshow(color,'gray')
plt.show()
