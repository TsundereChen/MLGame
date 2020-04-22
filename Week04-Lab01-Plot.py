import pickle
from os import path
import numpy as np
from matplotlib import pyplot as plt

def get_ArkanoidData(filename):
    file = open(filename,'rb')
    log = pickle.load(file)
    Frames = []
    Balls = []
    PlatformPos = []
    for sceneInfo in log:
        Frames.append(sceneInfo.frame)
        Balls.append([sceneInfo.ball_location[0], sceneInfo.ball_location[1]])

        PlatformPos.append(sceneInfo.board_location)

    frame_ary = np.array(Frames)
    frame_ary = frame_ary.reshape((len(Frames), 1))
    data = np.hstack((frame_ary, Balls, PlatformPos))
    return data

if __name__ == '__main__':
    filename = path.join(path.dirname(__file__), 'pickles', 'easy.1.calsol.pickle')
    data = get_ArkanoidData(filename)

    Frame = data[:,0]
    Ball_x = data[:,1]
    Ball_y = data[:, 2]
    Platform_x = data[:,3]
    Platform_y = data[:, 4]

    # plt.plot(Ball_x,Ball_y)
    plt.scatter(Ball_x,Ball_y)
    plt.figure()
    plt.scatter(Frame,Platform_x,c='BLUE',marker='.')
    plt.scatter(Frame, Ball_y, c='ORANGE', marker='x')
    plt.show()
