import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils
import math
from tensorflow.keras.models import model_from_json

def euclidean_distance(a,b):

    '''
        parametros: 
            a, b --> puntos a los que se quiere hallar la distancia
        
        retorno:
            devuelve la distancia euclidena que separa a y b

    '''
    return math.sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2)

def edge_delete(th):

    '''
        parametros:
            th -- > imagen binaria 

        retorno:
            digit --> imagen sin marcos 

    '''
    cnts = cv.findContours(th.copy(),
                           cv.RETR_EXTERNAL,
                           cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)

    h, w = th.shape

    if len(cnts) == 0:
        return np.zeros((h,w))

    c = max(cnts, key=cv.contourArea)
    mask = np.zeros(th.shape, dtype='uint8')
    cv.drawContours(mask, [c], -1, 255 , -1)

    
    error = cv.countNonZero(mask) / float(w*h)

    if error < 0.03:
        return np.zeros((h,w))

    digit = cv.bitwise_and(th, th, mask = mask)

    return digit

def preprocess(img):

    '''
        parametros:
            img --> imagen a color a procesar
        
        retorno:
            thresh --> imagen filtrada en blanco y negro

    '''

    ## Transforma la imagen a escala de grises
    gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

    ## Filtro blur para reducir el ruido
    blur = cv.GaussianBlur(gray, (11,11), 0)

    ## Binariza la imagen usando un filtro adaptativo
    thresh = cv.adaptiveThreshold(blur,255,
                            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv.THRESH_BINARY_INV,9,3)

    return thresh

## Carga la imagen
path = r"/home/ernesto/Documents/I_SI/sudoku/img/sudoku_3.jpg"
img = cv.imread(path)

## Preprosesamiento de la imagen
th = preprocess(img)

## Encuentra los contornos
cnts = cv.findContours(th, cv.RETR_EXTERNAL,
    cv.CHAIN_APPROX_SIMPLE)
cnts = imutils.grab_contours(cnts)
cnts = sorted(cnts, key=cv.contourArea, reverse=True)

points = None

for c in cnts:
    peri = cv.arcLength(c, True)
    approx = cv.approxPolyDP(c, 0.02 * peri, True)
    if len(approx) == 4:
        points = approx
        break

## Calculo del borde mas largo
bl, tl, tr, br = points.reshape((4,2))
s = int(max(euclidean_distance(bl,tl),
        euclidean_distance(tl,tr),
        euclidean_distance(tr,br),
        euclidean_distance(br,bl)))

a = points.reshape((4,2))

order = a[a[:,0].argsort()]

first = order[:2,:]
last = order[2:,:]
first = first[first[:,1].argsort()]
last = last[last[:,1].argsort()]
bl = first[0]
tl = first[1]
tr = last[0]
br = last[1]

## Se obtiene los puntos para la homografía
src = np.float32([bl,tl,tr,br])
dst = np.float32([[0,0],[0,s-1],[s-1,0],[s-1,s-1]])

## Calculo de la homografía
H = cv.getPerspectiveTransform(src,dst)

## Transformación de perspectiva
trans =  cv.warpPerspective(cv.cvtColor(img,cv.COLOR_BGR2GRAY),
                                        H,(s-1,s-1))
blur_var = cv.GaussianBlur(trans,(3,3),0)
_,trans = cv.threshold(blur_var,0,255,
                       cv.THRESH_BINARY_INV|cv.THRESH_OTSU)

# Abrir modelo pre-entrenado
json_file = open('CNN.json','r').read()
CNN = model_from_json(json_file)
CNN.load_weights('model.h5')

## Separar los 81 bloques
d = (s-1)/9
block = []
m = np.zeros((9,9))
r = int((s-1)*0.01)
for i in range(9):
    for j in range(9):
        ## Extrae la ROI para cada digito
        var = trans[int(i*d)+r:int((i+1)*d)-r,
                    int(j*d)+r:int((j+1)*d-r)]
        var = edge_delete(var)

        ## Guardar los numeros en la matriz
        b = cv.resize(var, (28,28))
        kernel = np.ones((2,2),np.uint8)
        b = cv.erode(b,kernel,iterations = 1).astype('uint8')
        _,b = cv.threshold(b,0,255,cv.THRESH_BINARY|cv.THRESH_OTSU)
        block.append(b)
        x = b.reshape(1,28,28,1).astype('float32')/255
        
        ## Identifica el digito
        if np.sum(b[5:23,5:23]) < 3:
            m[i,j] = -1
        else:
            m[i,j] = CNN.predict_classes(x)
