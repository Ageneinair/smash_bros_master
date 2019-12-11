import cv2 as cv
import yaml
from argparse import ArgumentParser
import numpy as np
from random import randint


def drawRegins(regionBoxes,image):
    img = cv.imread(image)

    def drawSingleRegin(d):
        for k, v in d.items():
            if isinstance(v, dict):
                drawSingleRegin(v)
            else:
                y1, x1, y2, x2 = v
                cv.rectangle(img, (x1, y1), (x2, y2), thickness=3,
                            color=(randint(0, 255), randint(0, 255), randint(0, 255)))

    drawSingleRegin(regionBoxes)
    cv.imwrite('drawRegions.png', img)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config", default='../config/evo.yaml', help="path to config")
    parser.add_argument("--image", default='../data/EVO/Evo_2014_One_minute_kill/Evo_2014_One_minute_kill_432.jpg', help="Path to image")

    opt = parser.parse_args()

    with open(opt.config) as f:
        config = yaml.load(f)

    drawRegins(config, opt.image)
