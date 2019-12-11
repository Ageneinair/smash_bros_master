from classifiers import *
import yaml
import cv2 as cv
import heapq
from time import time
from argparse import ArgumentParser

def timer(func):
    def wraper(*args, **kwargs):
        before = time()
        result = func(*args, **kwargs)
        after = time()
        print("elapsed: %f s." % (after - before))
        return result
    return wraper

@timer
def check_video(pgc, cs, cl, video_path):
    # open video file
    vc = cv.VideoCapture(video_path)

    # a priority que to confirm if we need check the current frame
    check_que = [(1, 0)]  # (time, classifier's index) No.0 is pgc
    heapq.heapify(check_que)

    time = 0
    rval=vc.isOpened()
    while rval:
        time += 1
        rval, frame = vc.read()
        if not rval:
            break

        while check_que[0][0]<= time:
            _, i = heapq.heappop(check_que)

            if i == 0: # check playground
                change_flag, next_time = cs[0].check(time, frame)
                if change_flag:
                    print('%02d:%02d' % (time//30//60, time//30%60), cl[0], cs[0].current_state)
                    if cs[0].current_state != 'no_gaming': # now is gaming, add other classifers into check que
                        for ind in range(1, len(cs)):
                            heapq.heappush(check_que, (time,ind))
                heapq.heappush(check_que, (next_time, 0))

            if i:
                change_flag, next_time = cs[i].check(time, frame)
                if change_flag and cs[i].current_state !='':
                    print('%02d:%02d' % (time//30//60, time//30%60), cl[i], cs[i].current_state)
                if cs[0].current_state != 'no_gaming':
                    if (i == 3 or i == 4) and change_flag and cs[i].current_state !='': # In every gaming we only check once player's name
                        continue
                    else:
                        heapq.heappush(check_que, (next_time, i))

            if cs[1].current_state is None or cs[2].current_state is None:
                cs[0].stable_time_check_time = 30
            else:
                cs[0].stable_time_check_time = min(cs[1].current_state[1], cs[2].current_state[1])*30

    vc.release()


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--config_path", default='./config/evo.yaml', help="path to config")
    parser.add_argument("--video_path", default='./data/EVO/Evo_2014_One_minute_kill.mp4', help="Path to image")
    parser.add_argument("--playground_model_path", default='./checkpoints/playground_model_1.h5', help="Path to model's weights")
    parser.add_argument("--stack_model_path", default='./checkpoints/stack_model_2.h5', help="Path to model's weights")

    opt = parser.parse_args()
    config_path = opt.config_path
    video_path = opt.video_path
    playground_model_path = opt.playground_model_path
    stack_model_path = opt.stack_model_path
    # config_path = './config/evo.yaml'
    # video_path = './data/EVO/Evo_2014_One_minute_kill.mp4'
    # config_path = './config/TBH.yaml'
    # video_path = './data/TBH/TBH8 SSBM - TSM _ Leffen (Fox) Vs. C9 _ Mang0 (Falco) - Smash Melee Winners Semis.mp4'

    playground_label_names = ['forest', 'fountain_of_dream', 'no_gaming', 'pokemon_gym', 'space', 'space_with_star']
    playground_size = 192
    stack_label_names = ['bg', 'falcon', 'fox', 'pikachu']
    stack_size = 32

    # open config file
    with open(config_path) as f:
        config = yaml.load(f)

    # set classifers
    pgc = PlaygroundClassifier(playground_model_path, playground_size,
                                playground_size, playground_label_names)
    sc1 = StackClassifier(stack_model_path, stack_size, stack_size,
                            stack_label_names, config['stacks']['player1'])
    sc2 = StackClassifier(stack_model_path, stack_size, stack_size,
                            stack_label_names, config['stacks']['player2'])
    tc = OpticalCharacterReader(config['time']['position'], config['time']['preprocessing_operations'])
    nc1 = OpticalCharacterReader(config['names']['player1'], config['names']['preprocessing_operations'])
    nc2 = OpticalCharacterReader(config['names']['player2'], config['names']['preprocessing_operations'])
    pc1 = OpticalCharacterReader(config['percentages']['player1'], config['percentages']['preprocessing_operations'])
    pc2 = OpticalCharacterReader(config['percentages']['player1'], config['percentages']['preprocessing_operations'])
    cl = ['Playground:', 'Player1\'s Character:', 'Player2\'s Character:', 'Player1\'s Name:',
        'Player2\'s Name:', 'Player1\'s Percentage:', 'Player2\'s Percentage:', 'Time']
    cs = [pgc, sc1, sc2]

    check_video(pgc, cs, cl, video_path)
