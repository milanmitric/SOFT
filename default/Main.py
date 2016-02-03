import cv2
import numpy as np
import pylab as pl
import functions
import matplotlib.pyplot as plt
import helperFunctions as hf

#keras - za neuronsku mrezu
from keras.datasets import mnist


from keras.utils import np_utils

# Ucitavanje slike
img = cv2.imread('sudokuIn3.jpg')

 #Testiranje prikaza
#cv2.imshow('image',img)
#cv2.waitKey(0)
#cv2.destroyAllWindows()

# Pretravanje u grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Binarizacija slike
thresh = cv2.adaptiveThreshold(gray,255,1,1,17,17)
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

batch_size = 128
nb_classes = 10
nb_epoch = 20

# the data, shuffled and split between tran and test sets
(X_train, y_train), (X_test, y_test) = mnist.load_data()



Temp_X_train = X_train
Temp_X_test = X_test

X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
Y_test = np_utils.to_categorical(y_test, nb_classes)

#ann = hf.create_ann()
#ann = hf.train_ann(ann,X_train,Y_train,X_test, Y_test)

#test = hf.prepare_for_ann(test)

#rezultat = ann.predict(np.array(test))
#alphabet = [0,1,2,3,4,5,6,7,8,9]
#print hf.display_result(rezultat,alphabet)

image = test[3]
tmp = []
tmparray = []
for i in range(len(image)):
    tmparray.append(image[i])
    if (i+1)%28==0:
        tmp.append(tmparray)
        tmparray = []


tmp = np.array(tmp)
cv2.imshow('image',tmp)
cv2.waitKey(0)
cv2.destroyAllWindows()

from PIL import Image
import pytesser
cv2.imwrite('tmp.png',tmp)
imagefile = Image.open('tmp.png')

tekst = pytesser.image_to_string(imagefile)
print tekst
