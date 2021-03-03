import numpy as np
import cv2
import time
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde

def func(theta, phi, cx, cy, cz):
	return np.float32([np.cos(theta), np.sin(theta)]) * cz / np.tan(-phi) + np.float32([cx, cy])

mu_cx = 1.0
mu_cy = 0.5
mu_cz = 0.8
sigma_cx = 0.1
sigma_cy = 0.1
sigma_cz = 0.2
mu_theta = np.radians(30)
mu_phi = np.radians(-45)
sigma_theta = np.radians(5)
sigma_phi = np.radians(3)

sample = 1000

start = time.time()

cx = np.random.normal(mu_cx, sigma_cx, sample)
cy = np.random.normal(mu_cy, sigma_cy, sample)
cz = np.random.normal(mu_cz, sigma_cz, sample)
theta = np.random.normal(mu_theta, sigma_theta, sample)
phi = np.random.normal(mu_phi, sigma_phi, sample)

results = func(theta, phi, cx, cy, cz)
mean = np.mean(results, axis=1)
std = np.std(results, axis=1)
kde = gaussian_kde(results)
#z = kde(results)

#plt.scatter(results[0], results[1], s=1, c=z)
#plt.show()

sigma_range = 3
x = np.linspace(mean[0] - std[0] * sigma_range, mean[0] + std[0] * sigma_range, 100)
y = np.linspace(mean[1] - std[1] * sigma_range, mean[1] + std[1] * sigma_range, 100)
xy = np.array(np.meshgrid(x, y))
heatmap = kde(xy.reshape((2,-1))).reshape(xy.shape[1:])[::-1]

print(time.time() - start)

plt.imshow(heatmap, cmap='jet')
plt.colorbar()
plt.show()
