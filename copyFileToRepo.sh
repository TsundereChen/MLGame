#!/bin/bash

rm -Rf ../mlgame-pingpong-ml-bot/saved_model
cp ./training.py ../mlgame-pingpong-ml-bot
cp ./validation.py ../mlgame-pingpong-ml-bot
cp -r ./log/* ../mlgame-pingpong-ml-bot/log
cp -r ./saved_model ../mlgame-pingpong-ml-bot
cp -r ./games/pingpong/ml/ml_play_NN.py ../mlgame-pingpong-ml-bot
