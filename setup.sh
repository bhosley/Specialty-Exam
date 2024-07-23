#!/bin/bash

# update environment's package manager
apt update && apt upgrade -y

# install python requirements
apt install -y python3-pip
python3 -m pip install --upgrade pip
#python3 -m pip install -r requirements.txt
# Due to dependency issues .txt method will not work
# Python libs must be installed in groups

pip install numpy==1.24.3
pip install torch
pip install swig
pip install box2d-py
pip install gymnasium[box2d]
pip install pettingzoo
pip install ray[rllib,tune,air]

# install WandB if it will be used
echo -e "\n\n"
read -p "Do you wish to use WandB (y/n)?" yesno
case $yesno in
    [Yy]* ) 
        pip install wandb
        wandb login
    ;;
    * ) 
    echo "If you change your mind, execute:"
    echo "pip install wandb"
    echo "wandb login"
    ;;
esac

printf "\nSetup complete"