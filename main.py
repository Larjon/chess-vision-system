import chess
import cv2
import chess.svg
import json
from cairosvg import svg2png
import numpy as np
import glob
import time
import chess


pp = 1

def nothing(X):
    pass


def thresold_calibreation(img):
    cv2.namedWindow("thresold_calibration")
    cv2.createTrackbar("thresold", "thresold_calibration", 0, 255, nothing)
    while True:
        t =  cv2.getTrackbarPos("thresold", "thresold_calibration")
        matrix,thresold = cv2.threshold(img,t,255,cv2.THRESH_BINARY)
        cv2.imshow("thresold",thresold)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            return t     


#Niwelowanie zniekształcen rybiego oka
def disortion(path, nr):
    # Wczytanie obrazka
    img = cv2.imread(path)

    # Współczynniki do korekcji efektu rybiego oka
    k1 = -0.065
    k2 = 1

    # Środek obrazka
    x0 = img.shape[1] / 2
    y0 = img.shape[0] / 2

    # Promień obrazka
    r = min(x0, y0)

    # Tworzenie macierzy przekształcenia
    map_x = np.zeros_like(img[:,:,0]).astype(np.float32)
    map_y = np.zeros_like(img[:,:,0]).astype(np.float32)

    for y in range(img.shape[0]):
        for x in range(img.shape[1]):
            dx = x - x0
            dy = y - y0
            distance = (dx ** 2 + dy ** 2) ** 0.5

            if distance == 0:
                new_x = x
                new_y = y
            else:
                theta = (distance / r) ** k1 * k2
                new_x = x0 + theta * dx
                new_y = y0 + theta * dy

            map_x[y,x] = new_x
            map_y[y,x] = new_y

    # Korekcja efektu rybiego oka
    corrected_img = cv2.remap(img, map_x, map_y, cv2.INTER_LINEAR)

    # Zapisanie nowego obrazka
    cv2.imwrite(f'snapshot_{nr}.jpg', corrected_img)
    # cv2.imshow('Niwelowanie efektu rybiego oka', corrected_img)
    # print("wcisnij dowolny klawisz")
    # cv2.waitKey(0)




def polozenie(path):
    # Wczytanie obrazu
    img = cv2.imread(path)

    # Konwersja do skali szarości
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binaryzacja
    thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY)[1]

    # Wyszukiwanie konturów
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # Sortowanie konturów według pola powierzchni (od największego do najmniejszego)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Wybór tylko 64 największych konturów (kwadratów)
    contours = contours[:64]

    # Rysowanie linii na obrazie
    h, w = img.shape[:2]
    sq_h, sq_w = h // 8, w // 8

    for i in range(8):
        for j in range(8):
            x1, y1 = j * sq_w, i * sq_h
            x2, y2 = (j + 1) * sq_w, (i + 1) * sq_h
            
            # Rysowanie linii pionowych
            cv2.line(img, (x1, y1), (x1, y2), (0, 255, 0), 2)
            
            # Rysowanie linii poziomych
            cv2.line(img, (x1, y1), (x2, y1), (0, 255, 0), 2)
            
            # Wycinanie obszaru odpowiadającego danemu kwadratowi
            square = thresh[y1:y2, x1:x2]
            
            # Wyszukiwanie konturu białego obiektu w kwadracie
            square_contours, _ = cv2.findContours(square, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            square_contours = sorted(square_contours, key=cv2.contourArea, reverse=True)
            if square_contours:
                # Znalezienie środka ciężkości konturu białego obiektu
                M = cv2.moments(square_contours[0])
                if M["m00"] != 0:
                    cx = int(M["m10"] / M["m00"]) + x1
                    cy = int(M["m01"] / M["m00"]) + y1
                    
                    # Wypisanie położenia białego obiektu na obrazie
                    cv2.putText(img, f'{cx}, {cy}', (x1 + 5, y2 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
                    print("pozycja x" , cx)
                    print("pozycja y" , cy)

    # Wyświetlenie obrazu
    cv2.imshow('image', img)
    cv2.waitKey(0)




def change_size(ww):
   #Ustawienie kamery i dopasowanie jej krawedzi
    img = cv2.imread(f'snapshot_{ww}.jpg')
    #zmiana wspolrzednych od lewego gornego rogu
    input_points = np.float32([[124,36],[524,26],[110,436],[541,441]])
    width = 600
    height = 600
    converated_points = np.float32([[0,0],[width,0],[0,height],[width,height]])
    matrix = cv2.getPerspectiveTransform(input_points, converated_points)
    img_output = cv2.warpPerspective(img,matrix,(width,height))
    cv2.imwrite(f'snapshot_{ww}.jpg', img_output)
    # cv2.imshow("oryginal", img_output)
    # cv2.imshow("Nowe perspektywa", img_output)
    # print("zmiana wielkosci obrazu wcisnij dowolny klawisz")
    # cv2.waitKey(0)



def znajdowanie_pola():
    # Wczytanie obrazu o wymiarach 600x600
    img = cv2.imread('tresh.jpg')
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Podzielenie obrazu na 64 kwadraty
    rows, cols = gray.shape
    sq_size = rows // 8
    for i in range(8):
        for j in range(8):
            x = i * sq_size
            y = j * sq_size
            if (i + j) % 2 == 0:
                cv2.rectangle(img, (x, y), (x + sq_size, y + sq_size), (0, 0, 0), -1)
            else:
                cv2.rectangle(img, (x, y), (x + sq_size, y + sq_size), (255, 255, 255), -1)

    # Znalezienie dwóch kwadratów z największą ilością białego pola
    white_areas = []
    for i in range(8):
        for j in range(8):
            x = i * sq_size
            y = j * sq_size
            roi = gray[y:y+sq_size, x:x+sq_size]
            white_areas.append((cv2.countNonZero(roi), (i, j)))

    white_areas.sort(reverse=True)
    top_two = white_areas[:2]
    top_squares = []
    for area in top_two:
        x, y = area[1]
        top_squares.append((x, y))
        
    # Zaznaczenie wybranych kwadratów na obrazie
    i = 0
    for square in top_squares:
        x = square[0] * sq_size
        y = square[1] * sq_size
        cv2.rectangle(img, (x, y), (x + sq_size, y + sq_size), (0, 255, 0), 2)
        # print("Square with highest white count: ", convert_to_chess_notation(x, y))
        
        if i == 0:
            pierwsze = convert_to_chess_notation(x, y)
            i = i + 1
        if i == 1:
            drugie = convert_to_chess_notation(x, y)    

    razem = pierwsze+drugie
    return razem
    # Wyświetlenie obrazu z zaznaczonymi kwadratami
    cv2.imshow('image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    # Wypisanie polozenia kwadratów
    # print("Najwięcej białego pola w kwadracie:", top_squares[0])
    # print("Następne najwięcej białego pola w kwadracie:", top_squares[1])

def convert_to_chess_notation(x, y):
    letters = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h']
    row = 8 - (y // 100)
    col = letters[x // 100]
    # print("x= ",x)
    # print("y= ",y)
    return col + str(row)


print ("Hello szachy")

capture = cv2.VideoCapture(0)
capture.set(cv2.CAP_PROP_FRAME_WIDTH, 600)
capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 600)
print('Nacisnij dowolny klawisz')
cv2.waitKey(0)
time.sleep(2)
_, frame = capture.read()
path1 = (f'test_{pp}.jpg')
time.sleep(1)
_, frame = capture.read()
path1 = (f'test_{pp}.jpg')
cv2.imwrite(path1, frame)
cv2.waitKey(0)
path = (f'snapshot_{pp}.jpg')
cv2.imwrite(path, frame)
print('Zapisano poczatkowe zdjęcie parti')
img = cv2.imread(path)
cv2.imshow("oryginal", img)
cv2.waitKey(0)

change_size(pp)
disortion(path,pp)
     
board = chess.Board()
oo = 1
while True:
    
    cv2.imshow("oryginal", img)
    if pp == oo:
        print("Nacisnij s jeśli zrobiłeś ruch")
        oo = oo + 1

    _, frame = capture.read()
    grayscale = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # print('Pętla sie wykonuje!')


    key = cv2.waitKey(1)
    if key == ord('s'):
            pp = pp + 1
            cv2.imwrite(f'snapshot_{pp}.jpg', frame)
            path = (f'snapshot_{pp}.jpg')
            change_size(pp)
            disortion(path,pp)
            print('Zapisano zdjęcie!')
            
            pp = pp - 1
            image1 = cv2.imread(f'snapshot_{pp}.jpg')
            image1_gray = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)



            pp = pp + 1
            image2 = cv2.imread(f'snapshot_{pp}.jpg')
            image2_gray = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)


            diff = cv2.absdiff(image1,image2)
            diff = cv2.resize(diff,(800,800))
            # cv2.imshow("diff", img)
            # cv2.waitKey(0)



            diff_gray = cv2.cvtColor(diff,cv2.COLOR_BGR2GRAY)
            # cv2.imshow("diff_gray", diff_gray)
            cv2.imwrite("Difference_GrayScale_image.jpg",diff_gray)
            print('Czekam na dowolny klawisz!')
            cv2.waitKey(0)



            #KALIBRACJA!!!!
            # value = thresold_calibreation(diff_gray)
            # matrix,thresold = cv2.threshold(diff_gray,value,255,cv2.THRESH_BINARY)
            # cv2.imshow("thresold",thresold)

            ret, thresh = cv2.threshold(diff_gray, 54, 255, cv2.THRESH_BINARY)          
            # cnts,_ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            # print(len(cnts))

            # wyświetlenie obrazu z zastosowaną funkcją threshold
            cv2.imshow('Thresholded Imageaaaaaaaaa', thresh)
            cv2.imwrite("tresh.jpg",thresh)
            # cv2.waitKey(0)
            sciezka = ('tresh.jpg')
            # polozenie(sciezka)
            posuniecie = znajdowanie_pola()
            
            print(board)     
            licznik = 0
            while not board.is_game_over():
                fen = board.fen()

                print(fen)
                print("przed petla moje posuniecie to =",posuniecie)
                try:
                    board.push_san(posuniecie)
                    break
                except ValueError:
                    print("Invalid move, try again.")
                    posuniecie = posuniecie[2:] + posuniecie[:2]
                    print("(po wprowadzeniu bledu) moje posuniecie to =",posuniecie)
                    if licznik == 2:
                        print("coś poszlo nie tak")
                        break
                    licznik = licznik + 1
                    continue
            
            print(board)     

            

    if key == ord('q'):
        break

capture.release()