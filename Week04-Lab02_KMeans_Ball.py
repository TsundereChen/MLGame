import pickle
import numpy as np
from os import path
from matplotlib import pyplot as plt
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn import metrics

def transformCommand(command):
    if command == 1:
        return 2
    elif command == -1:
        return 1
    else:
        return 0
    pass

def get_ArkanoidData(filename):
    Frames = []
    Balls = []
    Commands = []
    PlatformPos = []
    log = pickle.load((open(filename, 'rb')))
    for sceneInfo in log:
        Frames.append(sceneInfo.frame)
        Balls.append([sceneInfo.ball_location[0], sceneInfo.ball_location[1]])
        PlatformPos.append(sceneInfo.board_location)
        Commands.append(transformCommand(sceneInfo.board_movement))

    commands_ary = np.array([Commands])
    commands_ary = commands_ary.reshape((len(Commands), 1))
    frame_ary = np.array(Frames)
    frame_ary = frame_ary.reshape((len(Frames), 1))
    data = np.hstack((frame_ary, Balls, PlatformPos, commands_ary))
    return data


if __name__ == '__main__':
    filename = path.join(path.dirname(__file__), 'pickles', 'easy.1.calsol.pickle')
    data = get_ArkanoidData(filename)
    data=data[1::]
    Balls = data[:, 1:3]
    Balls_next = np.array(Balls[1:])
    vectors = Balls_next - Balls[:-1]
    direction=[]

    for i in range(len(data)-1):
        if(vectors[i,0]>0 and vectors[i,1]>0):
            direction.append(0) #向右上為0
        elif(vectors[i,0]>0 and vectors[i,1]<0):
            direction.append(1) #向右下為1
        elif(vectors[i,0]<0 and vectors[i,1]>0):
            direction.append(2) #向左上為2
        elif(vectors[i,0]<0 and vectors[i,1]<0):
            direction.append(3) #向左下為3

    direction = np.array(direction)
    direction = direction.reshape((len(direction),1))
    data = np.hstack((data[1:,:], direction))

    mask = [1, 2, 3, 6]
    X = data[:, mask]
    Y = data[:, -2]
    Ball_x = data[:,1]
    Ball_y = data[:,2]
    Direct = data[:,-1]

    x_train , x_test,y_train,y_test = train_test_split(X,Y,test_size=0.2)
    platform_predict_clf = KMeans(n_clusters=3, random_state=0).fit(x_train,y_train)        
    y_predict = platform_predict_clf.predict(x_test)
    print(y_predict)
    accuracy = metrics.accuracy_score(y_test, y_predict)
    print("Accuracy(正確率) ={:8.3f}%".format(accuracy*100))
    
    
    ax = plt.subplot(111, projection='3d')  
    ax.scatter(X[Y==0][:,0], X[Y==0][:,1], X[Y==0][:,3], c='#FF0000', alpha = 1)  
    ax.scatter(X[Y==1][:,0], X[Y==1][:,1], X[Y==1][:,3], c='#2828FF', alpha = 1)
    ax.scatter(X[Y==2][:,0], X[Y==2][:,1], X[Y==2][:,3], c='#007500', alpha = 1)
    
    plt.title("KMeans Prediction")    
    ax.set_xlabel('Ball_x')
    ax.set_ylabel('Ball_y')
    ax.set_zlabel('Direction')
        
    plt.show()

    with open('save/clf_KMeans_BallAndDirection.pickle', 'wb') as f:
        pickle.dump(platform_predict_clf, f)

