from data_loader import DataSetGenerator
import os
import tensorflow as tf
import time
import cv2 as cv
from ocr import ocr_region

def convert_resize_image(image, IMG_HEIGHT, IMG_WIDTH):
    return tf.expand_dims(tf.image.resize(tf.convert_to_tensor(image, dtype=tf.uint8), [IMG_HEIGHT, IMG_WIDTH])/255.0,0)

class PlaygroundClassifier():
    def __init__(self, model_path, IMG_HEIGHT, IMG_WIDTH, label_names):
        self.model = tf.keras.models.load_model(model_path)
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.label_names = label_names
        self.change_flag = 0
        self.current_state = None
        self.last_state = None
        self.cofidence_gap = 3
        self.stable_time_check_time = 30

    def check(self, time, image):
        img = convert_resize_image(image[:,:,::-1], self.IMG_HEIGHT, self.IMG_WIDTH)
        batch_pred = self.model.predict(img)
        self.current_state = self.label_names[tf.argmax(batch_pred[0])]

        if self.current_state != self.last_state:
            self.change_flag += 1
            if self.change_flag > self.cofidence_gap:
                self.last_state = self.current_state
                return True, time + self.stable_time_check_time # the state changed and the next check time is 1 second later
            return False, time + 3 # the state maybe change and need check next frame
        else:
            self.change_flag = 0
            return False, time + 30

    def check_all(self, time, image):
        img = convert_resize_image(image[:,:,::-1], self.IMG_HEIGHT, self.IMG_WIDTH)
        batch_pred = self.model.predict(img)
        self.current_state = self.label_names[tf.argmax(batch_pred[0])]
        print(time, self.current_state)
        return False, time + 1

class StackClassifier():
    def __init__(self, model_path, IMG_HEIGHT, IMG_WIDTH, label_names, stack_position):
        self.model = tf.keras.models.load_model(model_path)
        self.IMG_HEIGHT = IMG_HEIGHT
        self.IMG_WIDTH = IMG_WIDTH
        self.label_names = label_names
        self.stack_position = [stack_position['stack1'], stack_position['stack2'],
                            stack_position['stack3'], stack_position['stack4'], ]
        self.change_flag = 0
        self.current_state = None
        self.last_state = None
        self.cofidence_gap = 3
        self.stable_time_check_time = 30

    def check(self, time, image):
        for i in range(3,-1,-1):
            # check all stack positions
            y1, x1, y2, x2 = self.stack_position[i]
            img = convert_resize_image(image[:,:,::-1][y1:y2, x1:x2], self.IMG_HEIGHT, self.IMG_WIDTH)
            batch_pred = self.model.predict(img)
            current_stack = self.label_names[tf.argmax(batch_pred[0])]

            if current_stack != 'bg':
                self.current_state = (current_stack, i + 1)
                break

        if current_stack == 'bg':
            self.current_state = ('bg', 1)

        if self.current_state != self.last_state:
            self.change_flag += 1
            if self.change_flag > self.cofidence_gap:
                self.last_state = self.current_state
                return True, time + self.stable_time_check_time # the state changed
            return False, time + 3 # the state maybe change and need check next frame
        else:
            self.change_flag = 0
            return False, time + self.stable_time_check_time

    def check_all(self, time, image):
        print(time, end='')
        for i in range(3,-1,-1):
            # check all stack positions
            y1, x1, y2, x2 = self.stack_position[i]
            img = convert_resize_image(image[:,:,::-1][y1:y2, x1:x2], self.IMG_HEIGHT, self.IMG_WIDTH)
            batch_pred = self.model.predict(img)
            current_stack = self.label_names[tf.argmax(batch_pred[0])]
            print(current_stack, end='')
        print()
        return False, time + 1

class OpticalCharacterReader():
    def __init__(self, position, preprocessing_operations):
        self.y1, self.x1, self.y2, self.x2 = position
        self.preprocessing_operations = preprocessing_operations
        self.change_flag = 0
        self.current_state = None
        self.last_state = None
        self.cofidence_gap = 1
        self.stable_time_check_time = 30

    def check(self, time, image):
        img = image[self.y1:self.y2, self.x1:self.x2]
        text = ocr_region(img, self.preprocessing_operations)
        text = text.strip()
        self.current_state = text

        if self.current_state != self.last_state:
            self.change_flag += 1
            if self.change_flag > self.cofidence_gap:
                self.last_state = self.current_state
                return True, time + self.stable_time_check_time # the state changed and the next check time is 1 second later
            return False, time + 1 # the state maybe change and need check next frame
        else:
            self.change_flag = 0
            return False, time + self.stable_time_check_time


if __name__ == "__main__":
    model_path = './checkpoints/playground_model_1.h5'
    label_names = ['forest', 'fountain_of_dream', 'no_gaming', 'pokemon_gym', 'space', 'space_with_star']
    pgc = PlaygroundClassifier(model_path, 192, 192, label_names)
    IMG_DIR ='/Users/Xipeng/Desktop/Smash_Bros_Master/data/_playground/pokemon_gym/TBH8 SSBM - TSM _ Leffen (Fox) Vs_5714.jpg'
    img = cv.imread(IMG_DIR)

    print(pgc.check(0, img))
