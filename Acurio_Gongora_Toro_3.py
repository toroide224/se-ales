import numpy as n
import matplotlib.pyplot as p
from scipy.io import wavfile
from scipy.fft import fft, ifft, fftshift, ifftshift
import streamlit as s

dt = 0.001
T = n.arange(-1, 1 + 0.9 * dt, dt)

TA1 = n.arange(-1, -0.5, dt)
TA2 = n.arange(-0.5, 0, dt)
TA3 = n.arange(0, 0.5, dt)
TA4 = n.arange(0.5, 0.5 + 0.9 * dt, dt)
A1 = 1 - 4*(TA1 + 1)
A2 = 4*(TA1 + 0.5) - 1
A3 = 1 - 4*(TA1)
A4 = 4*(TA1 - 0.5) - 1
A = n.concatenate((A1, A2, A3, A4))

TC1 = n.arange(-1, -0.5, dt)
TC2 = n.arange(-0.5, 0.5, dt)
TC3 = n.arange(0.5, 0.5 + 0.9 * dt, dt)
C1 = (2 * n.pi * TC2 + 2 * n.pi)**2
C2 = (2 * n.pi * TC2)**2
C3 = (2 * n.pi * TC2 - 2 * n.pi)**2
C = n.concatenate((C1, C2, C3))

"""
# Señales y sistemas
## Series de fourier
Esta sección se encuentra incompleta
"""

@s.fragment()
def Series():
    Signal = s.selectbox("Elige la señal", ["3.6.1", "3.6.2", "3.6.3", "3.6.4"])
    ArmN = s.slider("Numero de armonicos", 1, 20)

"""
## Modulación de audio
"""

@s.fragment()
def AudioPr():
    F, X = wavfile.read("Dies_Irae.wav")
    X1 = 0.5*X[:,0] + 0.5*X[:,1]
    FC = s.slider("Frecuencia de portadora", 10000, 15000, step=500)

    t = n.arange(0, len(X1))/F
    XC1 = n.cos(2 * n.pi * FC * t)

    f = n.arange(-len(t)/2, len(t)/2) * F / len(t)
    X1F = fftshift(fft(X1))
    XC1F = fftshift(fft(XC1))
    
    Entradas = p.figure(figsize=(10,9))
    p.subplot(2, 1, 1)
    p.plot(t, X1)
    p.subplot(2, 1, 2)
    p.plot(f, (abs(X1F) / len(f)) ** 2)

    Portadoras = p.figure(figsize=(10,9))
    p.subplot(2, 1, 1)
    p.plot(t[0 : 30], XC1[0 : 30])
    p.subplot(2, 1, 2)
    p.plot(f, (abs(XC1F) / len(f)) ** 2)

    X1LF = X1F
    for k in range(len(f)):
        if abs(f[k]) > 2500:
            X1LF[k] = 0
    
    X1L = ifft(ifftshift(X1LF))

    Limitadas = p.figure(figsize=(10,9))
    p.subplot(2, 1, 1)
    p.plot(t, X1L)
    p.subplot(2, 1, 2)
    p.plot(f, (abs(X1LF) / len(f)) ** 2)

    X1M = X1L * XC1
    X1MF = fftshift(fft(X1M))

    Moduladas = p.figure(figsize=(10,9))
    p.subplot(2, 1, 1)
    p.plot(t, X1M)
    p.subplot(2, 1, 2)
    p.plot(f, (abs(X1MF) / len(f)) ** 2)

    X1D = X1M * XC1
    X1DF = fftshift(fft(X1D))

    Demoduladas = p.figure(figsize=(10,9))
    p.subplot(2, 1, 1)
    p.plot(t, X1D)
    p.subplot(2, 1, 2)
    p.plot(f, (abs(X1DF) / len(f)) ** 2)

    X1OF = X1DF
    for k in range(len(f)):
        if abs(f[k]) > 2500:
            X1OF[k] = 0

    X1O = ifft(ifftshift(X1OF))

    Filtradas = p.figure(figsize=(10,9))
    p.subplot(2, 1, 1)
    p.plot(t, X1O)
    p.subplot(2, 1, 2)
    p.plot(f, abs(X1OF / len(f))**2)

    Chose = s.radio("Señales a visualizar", ["Entrada" , "Filtrada", "Portadora", "Modulada", "Demodulada", "Recuperada"], horizontal=True)

    if Chose == "Entrada":
        s.pyplot(Entradas)
    if Chose == "Filtrada":
        s.pyplot(Limitadas)
    if Chose == "Portadora":
        s.pyplot(Portadoras)
    if Chose == "Modulada":
        s.pyplot(Moduladas)
    if Chose == "Demodulada":
        s.pyplot(Demoduladas)
    if Chose == "Recuperada":
        s.pyplot(Filtradas)    
AudioPr()

"""
## Modulación por cuadratura
"""

@s.fragment()
def QuadMod():
    F1 = s.slider("Frecuencia señal 1", 1000, 4000, step=500)
    F2 = s.slider("Frecuencia señal 2", 1000, 4000, step=500)
    FC = 100000
    delta = 1/(20 * FC)

    t = n.arange(0, 0.008 + 0.9 * delta, delta)
    X1 = n.cos(2 * n.pi * F1 * t)
    X2 = n.cos(2 * n.pi * F2 * t)
    XC1 = n.cos(2 * n.pi * FC * t)
    XC2 = n.sin(2 * n.pi * FC * t)

    f = n.arange(-len(t)/2, len(t)/2) / (delta * len(t))
    X1F = fftshift(fft(X1))
    X2F = fftshift(fft(X2))
    XC1F = fftshift(fft(XC1))
    XC2F = fftshift(fft(XC2))

    A = int(len(f) * (1/2 - 5000 * delta))
    B = int(len(f) * (1/2 + 5000 * delta))
    C = int(len(f) * (1/2 - 200000 * delta))
    D = int(len(f) * (1/2 + 200000* delta))
    
    Entradas = p.figure(figsize=(10,9))
    p.subplot(2, 2, 1)
    p.plot(t[0 : int(0.002/delta)], X1[0 : int(0.002/delta)])
    p.subplot(2, 2, 2)
    p.plot(f[A : B], (abs(X1F[A : B]) / len(f)) ** 2)
    p.subplot(2, 2, 3)
    p.plot(t[0 : int(0.002/delta)], X2[0 : int(0.002/delta)])
    p.subplot(2, 2, 4)
    p.plot(f[A : B], (abs(X2F[A : B]) / len(f)) ** 2)

    Portadoras = p.figure(figsize=(10,9))
    p.subplot(2, 2, 1)
    p.plot(t[0 : 40], XC1[0 : 40])
    p.subplot(2, 2, 2)
    p.plot(f[C : D], (abs(XC1F[C : D]) / len(f)) ** 2)
    p.subplot(2, 2, 3)
    p.plot(t[0 : 40], XC2[0 : 40])
    p.subplot(2, 2, 4)
    p.plot(f[C : D], (abs(XC2F[C : D]) / len(f)) ** 2)

    X1M = X1 * XC1
    X2M = X2 * XC2
    X1MF = fftshift(fft(X1M))
    X2MF = fftshift(fft(X2M))

    Moduladas = p.figure(figsize=(10,9))
    p.subplot(2, 2, 1)
    p.plot(t[0 : int(0.002/delta)], X1M[0 : int(0.002/delta)])
    p.subplot(2, 2, 2)
    p.plot(f[C : D], (abs(X1MF[C : D]) / len(f)) ** 2)
    p.subplot(2, 2, 3)
    p.plot(t[0 : int(0.002/delta)], X2M[0 : int(0.002/delta)])
    p.subplot(2, 2, 4)
    p.plot(f[C : D], (abs(X2MF[C : D]) / len(f)) ** 2)

    XM = X1M + X2M
    XMF = fftshift(fft(XM))

    Cuadratura = p.figure(figsize=(10,9))
    p.subplot(2, 1, 1)
    p.plot(t[0 : int(0.002/delta)], XM[0 : int(0.002/delta)])
    p.subplot(2, 1, 2)
    p.plot(f[C : D], (abs(XMF[C : D]) / len(f)) ** 2)

    X1D = 2 * XM * XC1
    X2D = 2 * XM * XC2
    X1DF = fftshift(fft(X1D))
    X2DF = fftshift(fft(X2D))

    Demoduladas = p.figure(figsize=(10,9))
    p.subplot(2, 2, 1)
    p.plot(t[0 : int(0.002/delta)], X1D[0 : int(0.002/delta)])
    p.subplot(2, 2, 2)
    p.plot(f, (abs(X1DF) / len(f)) ** 2)
    p.subplot(2, 2, 3)
    p.plot(t[0 : int(0.002/delta)], X2D[0 : int(0.002/delta)])
    p.subplot(2, 2, 4)
    p.plot(f, (abs(X2DF) / len(f)) ** 2)

    X1OF = X1DF
    X2OF = X2DF

    for k in range(len(f)):
        if abs(f[k]) > F1 + 500:
            X1OF[k] = 0
        if abs(f[k]) > F2 + 500:
            X2OF[k] = 0

    X1O = ifft(ifftshift(X1OF))
    X2O = ifft(ifftshift(X2OF))

    Filtradas = p.figure(figsize=(10,9))
    p.subplot(2, 2, 1)
    p.plot(t[0 : int(0.002/delta)], X1O[0 : int(0.002/delta)])
    p.subplot(2, 2, 2)
    p.plot(f[A : B], abs(X1OF[A : B] / len(f))**2)
    p.subplot(2, 2, 3)
    p.plot(t[0 : int(0.002/delta)], X2O[0 : int(0.002/delta)])
    p.subplot(2, 2, 4)
    p.plot(f[A : B], abs(X2OF[A : B] / len(f))**2)

    Chose = s.radio("Señales a visualizar", ["Entradas" , "Portadoras", "Moduladas", "Cuadratura", "Demoduladas", "Filtradas"], horizontal=True)

    if Chose == "Entradas":
        s.pyplot(Entradas)
    if Chose == "Portadoras":
        s.pyplot(Portadoras)
    if Chose == "Moduladas":
        s.pyplot(Moduladas)
    if Chose == "Cuadratura":
        s.pyplot(Cuadratura)
    if Chose == "Demoduladas":
        s.pyplot(Demoduladas)
    if Chose == "Filtradas":
        s.pyplot(Filtradas)    
QuadMod()

"""
## Modulación con gran portadora
"""

@s.fragment()
def LarC():
    IDM = s.selectbox("Indice de modulación", [0.7, 1, 1.2])
    A = s.slider("Amplitud (f = 7000)", 1, 3, step=0.1)
    B = s.slider("Amplitud (f = 4500)", 1, 3, step=0.1)
    C = s.slider("Amplitud (f = 1200)", 1, 3, step=0.1)

    FC = 500000
    F1 = 7000
    F2 = 4500
    F3 = 1200
    delta = 1/(20 * FC)

    t = n.arange(0, 0.002 + 0.9 * delta, delta)
    X1 = A * n.cos(2 * n.pi * F1 * t)
    X2 = B * n.cos(2 * n.pi * F2 * t)
    X3 = C * n.cos(2 * n.pi * F3 * t)
    X = X1 + X2 + X3
    XC = n.cos(2 * n.pi * FC * t)
    
    f = n.arange(-len(t)/2, len(t)/2) / (delta * len(t))
    N = int(len(f) * (1/2 - 10000 * delta))
    M = int(len(f) * (1/2 + 10000 * delta))

    XF = fftshift(fft(X))

    Entrada = p.figure(figsize=(10,9))
    p.subplot(2, 1, 1)
    p.plot(t, X)
    p.subplot(2, 1, 2)
    p.plot(f[N : M], abs(XF[N : M] / len(f))**2)

    XM = (X + abs(min)/IDM) * XC
    XMF = fftshift(fft(XM))

    Modulada = p.figure(figsize=(10,9))
    p.subplot(2, 1, 1)
    p.plot(t, XM)
    p.subplot(2, 1, 2)
    p.plot(f, abs(XMF / len(f))**2)

    XR1 = XM
    XR2 = XM
    for k in range(len(XM)):
        if XM[k] < 0:
            XR1[k] = 0
            XR2[k] = -XM[k]
    
    Rectificada = p.figure(figsize=(10,9))
    p.subplot(2, 1, 1)
    p.plot(t, XR1)
    p.subplot(2, 1, 2)
    p.plot(t, XR2)

    """
    #### Señal original
    """
    s.pyplot(Entrada)
    """
    #### Señal modulada mediante LC
    """
    s.pyplot(Modulada)
    """
    #### Rectificaciones de la señal
    Se muestran rectificaciones de media onda y de onda completa
    """
    s.pyplot(Rectificada)
LarC()
