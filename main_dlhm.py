# Main script for simulating realistic digital lensless holography
from dlhm import *
import plotly.express as px
import os
import numpy as np
import cv2 as cv
# Load a sample (convert to grayscale and normalize)
intensityImage = np.array(cv.imread('data/BenchmarkTarget.png', cv.IMREAD_GRAYSCALE)).astype(float) / 255
lambda_ = 532e-9  # Wavelength of the light


h_max = 350e-9

sample = np.exp(-1j * 2 * np.pi * (1.51 - 1) * h_max * intensityImage / lambda_)
# sample = 1-intensityImage

# Simulation parameters
L = 8e-3  # Distance from the source to the hologram plane
z = 2e-3  # Distance from the source to the sample's plane
W_c = 5.55e-3  # Width of the sensor
lambda_ = 532e-9  # Wavelength
dx_in = 1.85e-6


num = 0

folders = os.listdir("/home/spm061102/Documents/TDG/Raw dataset/cells")

for i in range(folders):
    data = os.listdir(f"{folders[i]}")
    for j in range(len(data)):
        img = np.array(cv.imread('data/BenchmarkTarget.png', cv.IMREAD_GRAYSCALE)).astype(float) / 255

        cf =  img * np.exp(-1j * 2 * np.pi * (1.51 - 1) * np.random.normal(0.1, 0.8) * img)

        holo = dlhm(cf, dx_in, L, z, W_c, dx_in, lambda_, x0=0, y0=0, NA_s=0.1)

        ph = np.angle(cf)
        ph = (ph - np.min(ph))/(np.max(ph) - np.min(ph))

        amp = np.abs(cf)
        amp = (amp - np.min(amp))/(np.max(amp) - np.min(amp))


        cv.imwrite()
        
        num += 1



# Call the dlhm function to simulate digital lensless holograms
holo = dlhm(sample, dx_in, L, z, W_c, dx_in, lambda_, x0=0, y0=0, NA_s=0.1)

# Display the simulated hologram
fig = px.imshow(holo, color_continuous_scale='gray')
fig.write_html('test.html')
