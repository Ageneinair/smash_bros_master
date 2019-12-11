import cv2 as cv
import os
import yaml

if __name__ == "__main__":
    ROOT_DIR = os.path.abspath("../")
    # "data/EVO/Evo_2014_One_minute_kill"
    # "data/TBH/TBH8 SSBM - TSM _ Leffen (Fox) Vs"
    FRAMES_DIR = os.path.join(ROOT_DIR, "data/TBH/TBH8 SSBM - TSM _ Leffen (Fox) Vs")
    PLAYER1_DIR = os.path.join(ROOT_DIR,"data/stacks/player1/")
    PLAYER2_DIR = os.path.join(ROOT_DIR,"data/stacks/player2/")
    os.makedirs(PLAYER1_DIR,exist_ok=True)
    os.makedirs(PLAYER2_DIR,exist_ok=True)

    config_path = '../config/TBH.yaml'
    with open(config_path) as f:
        config = yaml.load(f)

    ind = 0
    frames = os.listdir(FRAMES_DIR)
    for frame_name in frames:
        FRAME_PAT = os.path.join(FRAMES_DIR, frame_name)
        file_name = frame_name.split('.')[0]
        img = cv.imread(FRAME_PAT)

        for player in config['stacks']:
            for i, bbox in enumerate(config['stacks'][player]):
                y1, x1, y2, x2 = config['stacks'][player][bbox]
                if player == 'player1':
                    cv.imwrite(PLAYER1_DIR+file_name+'_'+str(i)+'.PNG', img[y1:y2, x1:x2])
                else:
                    cv.imwrite(PLAYER2_DIR+file_name+'_'+str(i)+'.PNG', img[y1:y2, x1:x2])
