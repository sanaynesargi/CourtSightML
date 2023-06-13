from gc import callbacks
from APIData import APIData, build_team_data_dict

import json
import numpy as np
import os
import tensorflow as tf
import time
import tensorflowjs as tfjs
import shutil
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import load_model


def get_dataset(player_name):
    with open(f"{player_name}_saved_gamedata.json", "r") as data:
        return json.load(data)
    

def process_data(raw_data):
    data = []
    labels = []
    
    for pred, res in raw_data:
        data.append([v for v in pred.values()])
        labels.append([v for v in res.values()])
        
    return np.array(data), np.array(labels)


def split(data, percent):
    a1 = []
    a2 = []
    python_data = data.tolist()
    p_len = round(percent * len(python_data))
    
    for i in range(p_len):
        a1.append(python_data[i])
    
    for i in range(p_len, len(python_data)):
        a2.append(python_data[i])

    return np.array(a1), np.array(a2)


def format_single_game(game):
    game = game.tolist()
    
    return np.array([game])


def create_directories(player_name):
    if not os.path.exists("models/"):
        os.mkdir("models/")
    if not os.path.exists(f"models/{player_name}/"):
        os.mkdir(f"models/{player_name}/")
    

def create_model():
    # Define the LSTM model using the Keras Sequential API
    model = Sequential()
    model.add(LSTM(128, input_shape=(1,13)))
    model.add(Dense(3))

    # Compile the model with mean squared error loss and Adam optimizer
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])
    
    return model


def train_model(train_data, train_labels, test_data, test_labels, player_name):
    combined = "".join(player_name.split(" "))
    if os.path.exists(f"models/{player_name}/trained_model/"):
        print("Found")
        return load_model(f"models/{player_name}/trained_model/{player_name}.h5")
    if os.path.exists(f"modelsJS/{combined}/trained_model/"):
        return None
    
    # Define Check Point code to save model weights
    path_checkpoint = os.path.join("models", player_name, f"{player_name}_cp.ckpt")
    callback = tf.keras.callbacks.ModelCheckpoint(filepath=path_checkpoint,
                                                 save_weights_only=True,
                                                 verbose=1)

    model = create_model()
    
    if os.path.exists(path_checkpoint):
        model.load_weights(path_checkpoint)
    else:
        # Train the model on the training data with 100 epochs, batch size of 1, and validation on the testing data
        model.fit(train_data, train_labels, epochs=100, batch_size=1, validation_data=(test_data, test_labels), callbacks=[callback])

    model.save(f"models/{player_name}/trained_model/{player_name}.h5")
    
    return model


def learn(player_name, raw_data):
    #Retrieve data related to player name
    create_directories(player_name)

    # Process the raw data and split it into data and labels
    data, labels = process_data(raw_data)

    # Split the data and labels into training and testing sets using a split ratio of 0.8
    train_data, test_data = split(data, 0.8)
    train_labels, test_labels = split(labels, 0.8)
    
    # Reshape the training and testing data to have the shape (number of samples, 1, number of features) 3D Formatting Required
    train_data = train_data.reshape(train_data.shape[0], 1, train_data.shape[1])
    test_data = test_data.reshape(test_data.shape[0], 1, test_data.shape[1])
    
    trained_model = train_model(train_data, train_labels, test_data, test_labels, player_name)
    
    if not trained_model:
        return None

    loss, acc = trained_model.evaluate(test_data, test_labels, verbose=2)
    print("Restored model, accuracy: {:5.2f}%".format(100 * acc))
    
    return trained_model
    

def train_players():
    players = []
    with open("players.json", "r") as player_data:
        players = [e["firstName"] + " " + e["lastName"] for e in json.load(player_data)]
        
    i = 0
    index = 452

    for player_name in players:
        comb = "".join(player_name.split(" "))
        print(player_name, comb, i) 
        if player_name in os.listdir("models/") or comb in os.listdir("modelsJS/"):
            i += 1
            continue
        
        try:
            #build_team_data_dict()
            test_class = APIData(player_name, 2023)
            
            training_raw_data = test_class.create_dataset()
            
            if training_raw_data:
                model = learn(player_name, training_raw_data)
            
            for fname in os.listdir("./"):
                if fname.startswith(player_name):
                    try:
                        os.remove(fname)
                    except PermissionError:
                        shutil.rmtree(fname)
            
            print(f"Finished Training {player_name}. {i}/{len(players)}")
            time.sleep(2)
        except Exception as e:
            print(e)
            os.popen(f"say Saved Index: {i}. Error: {str(e)[:20]}")
        
        i += 1
            

def load_player_model(player_name):
    path = f"./models/{player_name}/trained_model/{player_name}.h5"
    model = load_model(path, compile=False)
    model.compile(loss='mean_squared_error', optimizer='adam', metrics=["accuracy"])
    return model


def convert_models(players, org):
    # if os.path.exists(f"./modelsJS"):
    #     shutil.rmtree(f"./modelsJS")
    #     os.mkdir("./modelsJS")
    # else:
    #     os.mkdir("./modelsJS")    
        
    for player in players:
        if ".git" in player:
            continue
        #print(org)
        model = load_player_model(org[player])
        tfjs.converters.save_keras_model(model, f"./modelsJS/{player}")
        

def main():
    a = set(os.listdir("./models2"))
    b = set(os.listdir("./modelsJS"))
    
    for d in a.difference(b):
        os.popen(f"cp -r ./models2/{d} ./modelsJS/{d}")
    # if len(a.difference(b)) > 0:
    #     convert_models(a.difference(b), org)
    # else:
    #     print("Hooray!")
    


if __name__ == "__main__":
    main()
