#!/bin/sh

python3 -u trainRealNVP.py --batch_size 64 --width 32 --K 4 --num_steps 50 --plot_interval 10 --out_file /tmp/realNVP/test 2> testLog.txt
python3 -u trainOTFlow.py --batch_size 64 --width 32 --nt 4 --num_steps 50 --plot_interval 10 --out_file /tmp/OTFlow/test 2>> testLog.txt
python3 -u trainVAEmnist.py --batch_size 256 --num_epochs 2 --width_enc 8  --width_dec 8  --out_file /tmp/VAE/test 2>> testLog.txt
python3 -u trainDCGANmnist.py --batch_size 64 --num_steps 50 --width_disc 8  --width_dec 8 --plot_interval 10 --out_file /tmp/DCGAN/test 2>> testLog.txt
python3 -u trainWGANmnist.py --batch_size 64 --num_steps 50 --width_disc 8  --width_dec 8 --plot_interval 10 --out_file /tmp/WGAN/test 2>> testLog.txt
python3 -u trainDCGANmnist.py --batch_size 64 --num_steps 50 --width_disc 8  --width_dec 8 --plot_interval 10 --init_g /tmp/VAE/test-g.pt --out_file /tmp/DCGAN/test2 2>> testLog.txt
python3 -u trainWGANmnist.py --batch_size 64 --num_steps 50 --width_disc 8  --width_dec 8 --plot_interval 10 --init_g /tmp/VAE/test-g.pt --out_file /tmp/WGAN/test2 2>> testLog.txt

