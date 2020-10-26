import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils
import math
from tensorflow.keras.models import model_from_json
from tensorflow.keras.models import load_model
import time
from DFS import DFS

def euclidean_distance(a,b):

    '''
        Calcula la distancia euclideana entre dos puntos

        parametros: 
            a, b --> puntos a los que se quiere hallar la distancia
        
        retorno:
            devuelve la distancia euclidena que separa a y b
    '''
    
    return math.sqrt((a[1]-b[1])**2 + (a[0]-b[0])**2)

def edge_delete(th):

    '''
        Elimina los marcos externos de un recuadro

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
    digit = np.zeros((h,w))

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
        Hace el preprocesamiento de la imagen

        parametros:
            img --> imagen a color a procesar
        
        retorno:
            thresh --> imagen filtrada en blanco y negro
    '''

    ## Extrae el canal V del espacio HSV
    gray = cv.cvtColor(img,cv.COLOR_BGR2HSV)[:,:,2]

    ## Filtro blur para reducir el ruido
    blur = cv.GaussianBlur(gray, (11,11), 0)

    ## Binariza la imagen usando un filtro adaptativo
    thresh = cv.adaptiveThreshold(blur,255,
                            cv.ADAPTIVE_THRESH_GAUSSIAN_C,
                            cv.THRESH_BINARY_INV,9,3)

    return thresh

def rect_generate(img):

    '''
        Saca los 4 puntos externos del tablero de sudoku

        parametros:
            img --> imagen de un digito
        
        retorno:
            x,y,w,h --> puntos extremos del sudoku
    '''

    copy = img.copy()
    cnt,h = cv.findContours(copy,
                            cv.RETR_EXTERNAL, 
                            cv.CHAIN_APPROX_SIMPLE)
    cnt = cnt[0]
    x,y,w,h = cv.boundingRect(cnt)

    return x,y,w,h

## Carga la imagen
path = r"/home/ernesto/Documents/I_SI/sudoku/img/sudoku2.png"
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
trans =  cv.warpPerspective(cv.cvtColor(img,cv.COLOR_BGR2HSV)[:,:,2],
                                        H,(s-1,s-1))

plt.imshow(trans,'gray');plt.show()

blur_var = cv.GaussianBlur(trans,(1,1),0)

_,trans = cv.threshold(blur_var,0,255,
                       cv.THRESH_BINARY_INV|cv.THRESH_OTSU)

plt.imshow(trans,'gray');plt.show()

# Abrir modelo pre-entrenado
# json_file = open('CNN.json','r').read()
# CNN = model_from_json(json_file)
# CNN.load_weights('model.h5')
MODEL_FILE = 'model-classifier.h5'
CNN = load_model(MODEL_FILE)

## Separar los 81 bloques
d = (s-1)/9
m = np.zeros((9,9))
r = int((s-1)*0.01)

for i in range(9):
    for j in range(9):
        ## Extrae la ROI para cada digito
        roi = trans[int(i*d)+r:int((i+1)*d)-r,
                    int(j*d)+r:int((j+1)*d-r)]
     
        var = edge_delete(roi)

        h, w = roi.shape
        if np.sum(roi[r*3:h-r*3,r*3:w-r*3])== 0:
            m[i,j] = 0
        else:
            x, y, w, h = rect_generate(var)
            prueba = var[y:y+h,x:x+w]
            hip = int(math.sqrt(w**2 + h**2 + 30))
            hh = int((hip-h)/2)
            ww = int((hip-w)/2)
            mat = np.zeros((hip,hip)).astype('uint8')
            mat[hh:hh+h, ww:ww+w] = prueba
            inv = cv.bitwise_not(mat.copy(), var.copy())
        
            ## Guardar los numeros en la matriz
            scale = (cv.resize(inv, (28,28), interpolation = cv.INTER_AREA))
            norm = scale / 255
            x = norm.reshape(1,28,28,1).astype('float32')
            m[i,j] = CNN.predict(x, verbose = 0).argmax()  

dfs = DFS(m)
dfs.solve_sudoku(m)
print(m)