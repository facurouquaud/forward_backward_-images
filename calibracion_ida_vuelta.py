# -*- coding: utf-8 -*-
"""
Created on Thu Oct 23 09:55:05 2025

@author: Luis1
"""

import matplotlib.pyplot as plt
import numpy as np
import sys
import tifffile as tiff
from PIL import Image
import pandas as pd
from scipy.signal import find_peaks
plt.style.use(r"C:\Users\Luis1\Downloads\gula_style.mplstyle")
plt.rcParams["text.usetex"] = False
plt.rcParams["font.family"] = "serif"

path = "C:\\Users\\Luis1\\Downloads\\"

sys.path.append(path)
import matplotvanda as vd


def graficar_ida(x,y,imagen):
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')   # origin='lower' para que y vaya de abajo hacia arriba

    ax.set_title("(ida) ", fontsize=12, fontweight='bold')
    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')    
def graficar_vuelta(x,y,imagen):
    imagen = np.flip(imagen, axis=1)
    fig, ax = plt.subplots(constrained_layout=True)
    im = ax.imshow(imagen, cmap='inferno',
                   extent=[x.min(), x.max(), y.min(), y.max()],
                   origin='lower')   # origin='lower' para que y vaya de abajo hacia arriba
    
    ax.set_title("(Vuelta) ", fontsize=12, fontweight='bold')

    ax.set_xlabel("x [µm]")
    ax.set_ylabel("y [µm]")
    fig.colorbar(im, ax=ax, label="Número de fotones")
    ax.set_aspect('equal', adjustable='box')
    plt.show()


#generar el perfil de intensidad con la ida y la vuelta.
def guardar_imagen_tiff(file, imagen_ida, imagen_vuelta):
    tiff.imwrite( path + file + "_ida.tif", imagen_ida.astype(np.float32))
    tiff.imwrite( path + file + "_vuelta.tif", np.flip(imagen_vuelta, axis = 1).astype(np.float32))



def delta_ida_vuelta(file_name, px_size, dwell_time):
    datos = pd.read_csv(path + file_name + ".csv")
    x = datos["Distance_(_)"]*px_size
    y = datos["Gray_Value"]
    peaks, properties = find_peaks(y, prominence=0.05*np.max(y), distance=5)
    if len(peaks) >= 2:
        # ordenar por altura de pico
        top2_idx = np.argsort(y[peaks])[-2:]
        peaks = peaks[top2_idx]
        peaks = np.sort(peaks)  # ordenar de izquierda a derecha
    
        # distancia entre picos (en µm)
        separation = np.abs(x[peaks[1]] - x[peaks[0]])
    else:
        separation = np.nan
        print(" No se detectaron dos picos claros.")
    
    # --- Graficar ---
    fig, ax = plt.subplots(constrained_layout=True)
    ax.plot(x, y, color='mediumblue', lw=1.5)
    ax.plot(x[peaks], y[peaks], "ro")
    
    if not np.isnan(separation):
        plt.axvline(x[peaks[0]], color='r', ls='--', alpha=0.6)
        plt.axvline(x[peaks[1]], color='r', ls='--', alpha=0.6)
        plt.text(np.mean(x[peaks]), np.max(y)*0.9, f"Δx = {separation:.2f} µm",
                 ha='center', color='r', fontsize=11, fontweight='bold')
       
    
        # Trazar la línea entre los dos puntos
        plt.plot([x[peaks[0]], x[peaks[1]]], [y[peaks[1]], y[peaks[1]]], color = "red")
     
    plt.text(0, np.max(y),f"$V_s$: {px_size/dwell_time:.3f} µm/µs",
              ha='left', color='black', fontsize=12)
    ax.set_xlabel("Distancia (µm)")
    ax.set_ylabel("Intensidad")
    ax.legend()
    vd.gula_grid(ax)
    plt.show()
    print(f"Separación entre emisores: {separation:.3f} µm")
    
    
#hacemos la curva de calibración:

