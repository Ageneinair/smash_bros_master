import pytesseract
from PIL import Image
import cv2 as cv
import numpy as np
import random


def delete_small_spot(img, min_height, min_width):
    contours, hier = cv.findContours(img, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
    for cidx,cnt in enumerate(contours):
        (x, y, w, h) = cv.boundingRect(cnt)
        if h < min_height or w < min_width:
            img[y:y+h, x:x+w] *=0

def ocr_region(img, operations):
    # cv.imwrite('%d.png'%random.randint(1,200), img)
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    for operation, parameter in operations:
        if operation == 'inRange':
            img = cv.inRange(img, lowerb=np.array(parameter[0]), upperb=np.array(parameter[1]))
        if operation == 'bitwise_not':
            img = cv.bitwise_not(img)
        if operation == 'delete_small_spot':
            delete_small_spot(img, parameter[0], parameter[1])
        # cv.imshow('result', img)
        # k = cv.waitKey()
    
    # cv.imshow('result', img)
    # k = cv.waitKey()
    img = Image.fromarray(np.uint8(img))
    text = pytesseract.image_to_string(img)
    # if text.strip() != '': print(text)
    return text

if __name__ == "__main__":
    # image = Image.open('test.png')
    #
    # print(type(image))
    # text = pytesseract.image_to_string(image)
    # print(text)

    img = cv.imread('179.png')
    hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    print(hsv[19,63])

    lowerb = np.array([0,0,110])
    upperb = np.array([180,45,255])#[180,255,255]
    mask = cv.inRange(hsv, lowerb=lowerb, upperb=upperb)
    # mask = cv.bitwise_not(mask)
    kernel = cv.getStructuringElement(cv.MORPH_RECT,(3,3))
    mask = cv.morphologyEx(mask, cv.MORPH_OPEN, kernel)
    mask = cv.morphologyEx(mask, cv.MORPH_CLOSE, kernel)
    # delete_small_spot(mask, 50, 50)

    # cv.imshow('input', open_out)
    # open_out = Image.fromarray(np.uint8(open_out))
    # text = pytesseract.image_to_string(open_out)
    # print('after: ',text)
    mask = cv.bitwise_not(mask)
    cv.imshow('mask', mask)
    mask = Image.fromarray(np.uint8(mask))
    text = pytesseract.image_to_string(mask)
    print('before: ',text)

    k = cv.waitKey()
    cv.destroyAllWindows()
