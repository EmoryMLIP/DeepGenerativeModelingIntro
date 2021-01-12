#!/bin/sh

python3 -u trainRealNVP.py --batch_size 256 --width 128 --K 6 --num_steps 20000 --plot_interval 1000 --out_file results/realNVP/trainRealNVP | tee  results/realNVP/trainRealNVP.txt
python3 -u trainOTFlow.py --batch_size 256 --width 32 --nt 4 --num_steps 20000 --plot_interval 1000 --out_file results/OTFlow/trainOTFlow |tee  results/OTFlow/trainOTFlow
python3 -u trainVAEmnist.py --batch_size 64 --num_epochs 50 --width_enc 32  --width_dec 32 --out_file results/VAE/trainVAE  |tee results/VAE/trainVAE.txt
python3 -u trainDCGANmnist.py --batch_size 64 --num_steps 50000 --width_disc 32  --width_dec 32 --plot_interval 1000 --out_file results/DCGAN/trainDCGAN-randomInit  |tee results/DCGAN/trainDCGAN-randomInit.txt
python3 -u trainWGANmnist.py --batch_size 64 --num_steps 50000 --width_disc 32  --width_dec 32 --plot_interval 1000 --out_file results/DCGAN/trainWGAN-randomInit  |tee results/WGAN/trainWGAN-randomInit.txt
python3 -u trainDCGANmnist.py --batch_size 64 --num_steps 50000 --width_disc 32  --width_dec 32 --plot_interval 1000 --init_g results/VAE/trainVAE-g.pt --out_file results/WGAN/trainDCGAN-vaeInit  |tee results/DCGAN/trainDCGAN-vaeInit.txt
python3 -u trainWGANmnist.py --batch_size 64 --num_steps 50000 --width_disc 32  --width_dec 32 --plot_interval 1000 --init_g results/VAE/trainVAE-g.pt --out_file  results/WGAN/trainWGAN-vaeInit  |tee results/WGAN/trainWGAN-vaeInit.txt


