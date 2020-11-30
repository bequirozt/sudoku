import numpy as np
import cv2 as cv
import matplotlib.pyplot as plt
import imutils
import math
import pyautogui
# from tensorflow.keras.models import model_from_json
# import tensorflow.keras.models.load_model
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


def perspective_change(th):

    '''
        Transforma la imagen tomando los marcos del sudoku 

        parametros:
            th --> imagen para transformar

        retorno:
            s --> dimensiones de la imagen
            trans --> imagen transformada y recortada
    '''

    ## Encuentra los contornos
    cnts = cv.findContours(th, cv.RETR_EXTERNAL,
        cv.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    cnts = sorted(cnts, key=cv.contourArea, reverse=True)

    points = None

    ## Extrae los 4 puntos del marco del sudoku
    for c in cnts:
        peri = cv.arcLength(c, True)
        approx = cv.approxPolyDP(c, 0.02 * peri, True)
        if len(approx) == 4:
            points = approx
            break

    ## Calculo del borde mas largo
    bl, tl, tr, br = points.reshape((4,2))

    ## Calculo de la distancia euclidiana mayor
    s = int(max(euclidean_distance(bl,tl),
                euclidean_distance(tl,tr),
                euclidean_distance(tr,br),
                euclidean_distance(br,bl)))

    a = points.reshape((4,2))

    order = a[a[:,0].argsort()]

    ## Organiza los puntos de la imagen
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

    return s, trans

def fill_image(m,k, num_array, img):

    '''
        Rellena los espacios del sudoku con los numeros que resuelven el sudoku

        parametros:
            m --> matriz con el sudoku resuelto
            k --> matriz con el sudoku sin resolver
            num_array --> lista de numeros para pintar sobre la imagen
            img --> imagen del sudoku

        retorno:
            output --> imagen con los numeros superpuestos
    '''

    for i in range(9):
        for j in range(9):
            if k[i,j] == 0:
                predict = int(m[i,j])
                img[int(i*d):int((i+1)*d),
                    int(j*d):int((j+1)*d)] = num_array[predict-1]
    output = img
    return output

##----------------------------------------------------------------------------------##
##-------------------------------Inicio del programa--------------------------------##
##----------------------------------------------------------------------------------##

if __name__ == "__main__":

    ## Carga la imagen
    # path = r"img/sudoku.png"
    # img = cv.imread(path)
    img = np.array(pyautogui.screenshot())

    ## Preprosesamiento de la imagen
    th = preprocess(img)

    ## Cambia la perspectiva y la recorta en la zona de interes
    s, trans = perspective_change(th)

    blur_var = cv.GaussianBlur(trans,(1,1),0)

    _,otsu = cv.threshold(trans,0,255,
                        cv.THRESH_BINARY_INV|cv.THRESH_OTSU)


    # Abrir modelo pre-entrenado
    # json_file = open('CNN.json','r').read()
    # CNN = model_from_json(json_file)
    # CNN.load_weights('model.h5')
    MODEL_FILE = 'model-classifier.h5'
    CNN = load_model(MODEL_FILE)

    ## Separa los 81 bloques
    d = int((s-1)/9)
    m = np.zeros((9,9))
    r = int((s-1)*0.01)

    ## Carga los digitos del 1 al 9
    num_array = []
    for n in range(0,9): 
        digit = cv.imread('nums/' + str(n+1) + '.png',0)
        # plt.imshow(digit)
        # plt.show()
        num_array.append(cv.resize(digit,(int(d),int(d))))

    ## Itera sobre cada bloque del sudoku
    for i in range(9):
        for j in range(9):
            ## Extrae la ROI para cada digito
            roi = otsu[int(i*d)+r:int((i+1)*d)-r,
                        int(j*d)+r:int((j+1)*d-r)]

            ## Elimina los bordes de la imagen
            roi = edge_delete(roi)
            h, w = roi.shape
            if np.sum(roi[r*3:h-r*3,r*3:w-r*3]) == 0:
                m[i,j] = 0
            else:

                x, y, w, h = rect_generate(roi)
                digit_cut = roi[y:y+h,x:x+w]
                hip = int(math.sqrt(w**2 + h**2 + 30))
                hh = int((hip-h)/2)
                ww = int((hip-w)/2)
                mat = np.zeros((hip,hip)).astype('uint8')
                mat[hh:hh+h, ww:ww+w] = digit_cut
                inv = cv.bitwise_not(mat.copy(), roi.copy())
            
                ## Guardar los numeros en la matriz
                scale = (cv.resize(inv, (28,28),
                                   interpolation = cv.INTER_AREA))
                norm = scale / 255
                x = norm.reshape(1,28,28,1).astype('float32')
                m[i,j] = CNN.predict(x, verbose = 0).argmax() 
                
    output = trans.copy()
    k = m.copy()

    plt.imshow(trans,'gray')
    plt.show()

    dfs = DFS(m)
    dfs.solve_sudoku()

    output = fill_image(m,k,num_array,output)
    result = cv.addWeighted(trans,0.7,output,0.2,0)

    plt.imshow(result,'gray')
    plt.show()