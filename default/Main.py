import cv2
import numpy as np
import pylab as pl
import functions
import helperFunctions as hf
# Ucitavanje slike
img = cv2.imread('sudokuIn2.png')

# Testiranje prikaza
# cv2.imshow('image',img)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Pretravanje u grayscale
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
# Binarizacija slike
thresh = cv2.adaptiveThreshold(gray,255,1,1,11,15)


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
                    np.array([0.0,0.0] ,np.float32) + np.array([144,0], np.float32),
                    np.array([0.0,0.0] ,np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([0.0,144], np.float32),
                    np.array([0.0,0.0] ,np.float32) + np.array([144,144], np.float32),
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
for i in range(len(indexes_numbers)):
    ind = indexes_numbers[i]
    if (i%5)==0 and i!=0:
        width=width+1
    axarr[i%5, width].imshow(cv2.resize(functions.sudoku[ind, :].reshape(functions.IMAGE_WIDTH,functions.IMAGE_HEIGHT), (functions.IMAGE_WIDTH*5,functions.IMAGE_HEIGHT*5)), cmap=pl.gray())
    axarr[i%5, width].axis("off")
