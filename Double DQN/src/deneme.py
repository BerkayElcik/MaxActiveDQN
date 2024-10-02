
import pandas as pd
import os
import numpy as np
import matplotlib.pyplot as plt
from decimal import Decimal


path_freq="data_ITU/freqs_0.5_1.csv"
path_loss="data_ITU/loss_matrix_0.5_1.csv"
path_noise="data_ITU/noise_matrix_0.5_1.csv"


freq_pd=pd.read_csv(path_freq,header=None)
loss_pd=pd.read_csv(path_loss,header=None)
noise_pd=pd.read_csv(path_noise,header=None)

print(freq_pd.head())
print(loss_pd.head())


print("max-min noise pd")
print(noise_pd.to_numpy().max())

print(noise_pd.to_numpy().min())

print("max-min loss pd")
print(loss_pd.to_numpy().max())

print(loss_pd.to_numpy().min())



print(f"{Decimal(loss_pd.to_numpy().max()):.2E}")

print(f"{Decimal(loss_pd.to_numpy().min()):.2E}")


freqs_array=freq_pd.to_numpy()
print(freqs_array)


print(freq_pd.diff(axis=1))

n_games=55
starting_eps=1
ending_eps=0.01
no_activity=4500
n_training_steps=n_games*no_activity

dec_step_of_epsilon=(starting_eps-ending_eps)/(n_training_steps*0.4)

print(dec_step_of_epsilon)