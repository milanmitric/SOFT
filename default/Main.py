import cv2
import numpy as np
import pylab as pl
import functions
import matplotlib.pyplot as plt
import helperFunctions as hf

#keras - za neuronsku mrezu
from keras.datasets import mnist

from PIL import Image
import pytesser
import re
from keras.utils import np_utils

# Ucitavanje slike
img = cv2.imread('sudokuIn3.jpg')

 #Testiranje prikaza
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Pretravanje u grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Binarizacija slike
thresh = cv2.adaptiveThreshold(gray,255,1,1,15,17)
#cv2.imshow('image',thresh)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Pronadji konture
_ ,contours, hierarchy = cv2.findContours(thresh,
                                        cv2.RETR_TREE,
                                        cv2.CHAIN_APPROX_SIMPLE)

# Sirina i velicina slike
height,width =  img.shape[:2]
print("Height: " + str(height) + " width: " + str(width))
# Kopiraj sliku
img_candidates = img.copy()

# Najveci pravougaonik, za pronalazak sudoku
size_rectangle_max = 0;
# Indeks najvece konture
index = 0
for i in range(len(contours)):
        # Aproksimiraj konture na poligon
        approximation = cv2.approxPolyDP(contours[i],4,True)

        # Da li poligon ima 4 strane?
        if (not (len(approximation) == 4)):
                continue

        # Da li je konveksan?
        if (not cv2.isContourConvex(approximation)):
                continue;
        # Povrsina poligona
        size_rectangle = cv2.contourArea(approximation)

        # Sacuvaj najveci
        if size_rectangle > size_rectangle_max and i != 0:
                index = i
                size_rectangle_max = size_rectangle
                big_rectangle = approximation




# Prikazi trenutno selektovani sudoku
approximation = big_rectangle
for i in range (len(approximation)):
        cv2.line(img_candidates,
                 (big_rectangle[(i%4)][0][0], big_rectangle[(i%4)][0][1]),
                 (big_rectangle[((i+1)%4)][0][0], big_rectangle[((i+1)%4)][0][1]),
                 (255, 0, 0), 2)


# Tacke u remap
points1 = np.array([
                    np.array([0.0,0.0] ,np.float32) + np.array([252,0], np.float32),
                    np.array([0.0,0.0] ,np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([0.0,252], np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([252,252], np.float32),
                    ],np.float32)
outerPoints = functions.getOuterPoints(approximation)
points2 = np.array(outerPoints,np.float32)

# Transformisi matricu
pers = cv2.getPerspectiveTransform(points2,  points1 );

warp = cv2.warpPerspective(img_candidates, pers, (functions.SUDOKU_SIZE*functions.IMAGE_HEIGHT, functions.SUDOKU_SIZE*functions.IMAGE_WIDTH));
# Posto slika dobijena ovako bude reflektovana potrebno je vratiti
warp = cv2.flip(warp,1)
warp_gray = cv2.cvtColor(warp, cv2.COLOR_BGR2GRAY)
# Remapiraj sliku

#cv2.imshow('image',warp_gray)
#cv2.waitKey(0)
#cv2.destroyAllWindows()


index_subplot=0
n_numbers=0
indexes_numbers = []

for i in range(functions.SUDOKU_SIZE):
    for j in range(functions.SUDOKU_SIZE):
        if functions.Recognize_number(i, j,warp_gray)==1:
            if (n_numbers%5)==0:
                index_subplot=index_subplot+1
            indexes_numbers.insert(n_numbers, i*9+j)
            n_numbers=n_numbers+1

#create subfigures
f,axarr= pl.subplots(index_subplot,5)

width = 0;
test = []
for i in range(len(indexes_numbers)):
    ind = indexes_numbers[i]
    test.append(functions.sudoku[ind])




rows = np.vsplit(warp_gray,9);
# Matrica brojeva
sudokuMatrix = np.zeros((81,),dtype=np.int);
# Brojac pozicija pocetnih brojeva
k = 0
# Niz pocetnih brojeva
startNums = []

tmp = []
tmparray = []
for j in range(len(test)):
   image = test[j]
   for i in range(len(image)):
        tmparray.append(image[i])
        if (i+1)%28==0:
            tmp.append(tmparray)
            tmparray = []
   tmp = np.array(tmp)
#   tmp = hf.erode(hf.dilate(tmp))
   cv2.imwrite('tmp.png',tmp)
   imagefile = Image.open('tmp.png')
   tekst = pytesser.image_to_string(imagefile)
   tekst = tekst.translate(None,'\n')
   tekst = tekst.translate(None,',')
   if tekst == '\"I':
       tekst = '1'
   elif tekst == 'Z':
       tekst = '2'
   elif tekst == '`I':
       tekst = '1'
   startNums.append(int(tekst))
   tmp = []

for k in range(len(indexes_numbers)):
    sudokuMatrix[indexes_numbers[k]] = startNums[k]

sudokuFinal = []
sudokuTmp = []

for i in range(len(sudokuMatrix)):
    sudokuTmp.append(sudokuMatrix[i])
    if (i+1)%9==0:
        sudokuFinal.append(sudokuTmp)
        sudokuTmp = []


sudokuMatrix  = sudokuFinal
print sudokuMatrix;

import solveSudoku as s
s.solveSudoku(sudokuMatrix)
print sudokuMatrix

font = cv2.FONT_HERSHEY_SIMPLEX
for i in range(len(sudokuMatrix)):
    for j in range(len(sudokuMatrix[i])):
        if not(indexes_numbers.__contains__(i*9+j)):
            #print str(sudokuMatrix[i][j])

            cv2.putText(warp,str(sudokuMatrix[i][j]),((j)*28, (i+1)*27 ),font,0.6,(0,0,0),1)


cv2.imshow("img",warp)
cv2.waitKey(0)

