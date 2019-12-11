import cv2 as cv
import os

def save_img():

    gap = 3
    VIDEO_DIR = '../data/TBH/'
    videos = os.listdir(VIDEO_DIR)
    print(videos)
    for video_name in videos:
        print(video_name)
        file_name = video_name.split('.')[0]
        folder_name = VIDEO_DIR + file_name
        os.makedirs(folder_name,exist_ok=True)
        vc = cv.VideoCapture(VIDEO_DIR+video_name)
        c = 0
        i = 0
        rval=vc.isOpened()

        while rval:
            c += 1
            rval, frame = vc.read()
            if c % gap != 0:
                continue
            i += 1

            PIC_DIR = folder_name+'/'
            if rval:
                cv.imwrite(PIC_DIR + file_name + '_' + str(i) + '.jpg', frame)
                cv.waitKey(1)
            else:
                break

        vc.release()

if __name__ == "__main__":
    save_img()
