# Smash Bros Gaming Stream Understander
This is a repo for understanding gaming stream of Smash Bros on Python 3, TensorFlow. The Understander takes a  Smash Bros gaming video as input, and automatically detect 1) gaming or not gaming, 2) playground type, 3) number of stacks of every player, 4) character of every player, 5) player's name, 6) gaming time, 7) percentages of every player.

## Installation
1. Clone this repository
2. Install dependencies
   ```bash
   pip3 install -r requirements.txt
   ```

## DEMO
   ```bash
    python3 run.py --config_path <path_to_config_file> --video_path <path_to_video_path>
   ```
### DEMO for [EVO 2014 clip](https://www.youtube.com/watch?v=bj7IX18ccdY)
   ```bash
    00:00 Playground: no_gaming
    00:05 Playground: fountain_of_dream
    00:05 Player1's Character: ('fox', 4)
    00:05 Player2's Character: ('pikachu', 4)
    00:19 Player1's Character: ('fox', 3)
    00:30 Player1's Character: ('fox', 2)
    00:53 Player1's Character: ('fox', 1)
    01:03 Player1's Character: ('bg', 1)
    01:04 Playground: no_gaming
    elapsed: 22.168249 s.
   ```

## CNN model
The playground classifier and stack classifier are pre-trained model of CNN. The data is from these three Youtube videos: [EVO 2014 clip](https://www.youtube.com/watch?v=bj7IX18ccdY), [TBH 8 clip](https://www.youtube.com/watch?v=FhO9zbjewfs), and [SS 7 clip](https://www.youtube.com/watch?v=Ns85L2lCWBI).

### Dataset
You can download the labeled data from here: [dataset for playground](https://drive.google.com/drive/folders/1PBp97KZfAhjnTJfKeZdQ1kja5L6mxao9?usp=sharing), [dataset for stack](https://drive.google.com/drive/folders/1GEQrvz48L3LdMJl2LN-vQ7VAkViptsXH?usp=sharing).

### Training Process
The models are trained in Google Colab with GPU, you can check the training process from [here for playground](https://github.com/Ageneinair/smash_bros_master/blob/master/colab_training/playground_classifier.ipynb) and [here for stack](https://github.com/Ageneinair/smash_bros_master/blob/master/colab_training/stack_classifer.ipynb).

## Feature Match
Also the repo has propose a method to find the region of interest by matching some specific features in the video (not write in pipeline yet), which can make the pipeline can generate the config file automatically. You can check the training from [here](https://github.com/Ageneinair/smash_bros_master/blob/master/feature_match.ipynb).
