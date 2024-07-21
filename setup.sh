#!/bin/bash

# update environment's package manager
apt update && apt upgrade -y
python3 -m pip install --upgrade pip

# install python requirements
python3 -m pip install -r requirements.txt

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

echo -e "\nSetup complete"