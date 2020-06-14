import pickle
import os
import numpy as np

path = os.getcwd()
path = os.path.join(path,"games","RacingCar","log")

allFile = os.listdir(path) # load log file
data_set = []


for file in allFile:
    with open(os.path.join(path,file),"rb") as f:
        data_set.append(pickle.load(f)) # load data in data_set

x = np.array([1, 2, 3])  # feature
y = np.array([0]) # label

for i, data in enumerate(data_set): 
    PlayerCar_x = []  
    PlayerCar_y = []
    Velocity = [] # feature
    Activity = [] # the label which we give it
    ComputerCar_lane = [] # feature : to record the computer car x
    Difference_y = []  # feature : to record the y_difference of computer car and player's car
    
    for scene_info in data["scene_info"][1::]: # start from the frame 1 
        ComputerCarDif_x = []
        ComputerCarDif_y = []
        ComputerCar_x = []
        ComputerCar_y = []
        for car in scene_info["cars_info"]:
            if car["id"] == 0:  # the player's car information          
                PlayerCar_x.append(car["pos"][0])
                PlayerCar_y.append(car["pos"][1])
                Velocity.append(car["velocity"])
                p_x = car["pos"][0]
                p_y = car["pos"][1]
                lane = car["pos"][0]//70

            else:
                ComputerCar_x.append(car["pos"][0])
                ComputerCar_y.append(car["pos"][1])
                ComputerCarDif_x.append(car["pos"][0]-p_x)
                ComputerCarDif_y.append(car["pos"][1]-p_y)
        
        for i in range(len(ComputerCarDif_x)):                    
            # ComputerCar_x // 70 means the computer car's lane
            if ComputerCar_x[i] //70 == lane and ComputerCarDif_y[i] <= 250 and ComputerCarDif_y[i] >0:
                # if computer car's lane is equal to player's lane and the y_difference is bewteen 0 and 250 means the forward site exist cars
                Activity.append(1110) # append 1110 to label ["BRAKE"]
                Difference_y.append(ComputerCarDif_y[i]) 
                ComputerCar_lane.append(ComputerCar_x[i]) # record the information of the moment which we add the label
                break
    
            elif ComputerCar_x[i]//70 == lane+1 and ComputerCarDif_y[i]<=80 and ComputerCarDif_y[i]>=-80:
                # if computer car's lane is in the right site of player's lane and the y_difference is bewteen -80 and 80 means the right site exust cars
                Activity.append(110) # append 110 to label ["MOVE_LEFT"]
                Difference_y.append(ComputerCarDif_y[i])
                ComputerCar_lane.append(ComputerCar_x[i]) # record the information of the moment which we add the label
                break
            
            elif ComputerCar_x[i]//70 == lane-1 and ComputerCarDif_y[i]<=80 and ComputerCarDif_y[i]>=-80:
                # if computer car's lane is in the left site of player's lane and the y_difference is bewteen -80 and 80 means the left site exust cars
                Activity.append(10) # append 10 to label ["RIGHT"]
                Difference_y.append(ComputerCarDif_y[i])
                ComputerCar_lane.append(ComputerCar_x[i]) # record the information of the moment which we add the label
                break
                                          
            elif ComputerCar_x[i] //70 == lane and ComputerCarDif_y[i] >=250:
                # if computer car's lane is equal to player's lane and the y_difference is more than 250 means the player's car can speed up
                Activity.append(0) # append 0 to label ["SPEED"]
                Difference_y.append(ComputerCarDif_y[i])
                ComputerCar_lane.append(ComputerCar_x[i]) # record the information of the moment which we add the label
                break
            
            if (i == len(ComputerCarDif_x)-1):
                # if all the computer's cars is not match the prevoius condition then means the player's car can speed up 
                Activity.append(0)
                Difference_y.append(ComputerCarDif_y[i])
                ComputerCar_lane.append(ComputerCar_x[i]) # record the information of the moment which we add the label
                break

#%%

    ComputerCar_lane = np.array(ComputerCar_lane[:len(Activity)]).reshape((len(Activity), 1))
    Difference_y = np.array(Difference_y[:len(Activity)]).reshape((len(Activity), 1))
    Velocity = np.array(Velocity[:len(Activity)]).reshape((len(Activity), 1))
    # reshape the feature and label

    x = np.vstack((x, np.hstack((ComputerCar_lane, Difference_y, Velocity,))))
    y = np.hstack((y, np.array(Activity)))
    # stack the feature and label

#%% training model
from sklearn.tree import DecisionTreeRegressor  
from sklearn.neural_network import MLPClassifier

x = x[1::] #remove [1, 2, 3]
y = y[1::] #remove [0]

#model = DecisionTreeRegressor(max_depth = 10) # you can set any max_depth
#model.fit(x, y)

model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=200)
model.fit(x, y)


with open('games/RacingCar/ml/save/model.pickle', 'wb') as f:
    pickle.dump(model, f)             