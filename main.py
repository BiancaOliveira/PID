import cv2
import numpy as np
import matplotlib.pyplot as plt
import math
import pandas as pd
from sklearn.preprocessing import normalize
import pickle
from numpy import array
from sklearn.preprocessing import StandardScaler
import sys
import os
import shutil



nome_arquivo = sys.argv[1]
imagem = cv2.imread(nome_arquivo,0)

#Colocando bordas na imagem
imagem = cv2.copyMakeBorder(imagem, 50, 50, 50, 50, cv2.BORDER_REPLICATE)
imagem_original = imagem

#Borra bastante
imagem = cv2.GaussianBlur(imagem,(75,75),0)


#Limiar binary invertido + limiar de OTSU o método de Otsu determina um valor de limite global ideal a partir do histograma da imagem

limiar, imagem_limiar = cv2.threshold(imagem,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU )

imagem_limiar_255 = imagem_limiar*255
imagem_limiar_255 = imagem_limiar_255.astype(np.uint8)

# Encontrando os contornos na imagem
contornos, void = cv2.findContours(imagem_limiar, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

#Aqui temos o numero de objetos na imagem, que vai ser igual ao numero de bordas encontradas
size = len(contornos)

aux = 1
df = pd.DataFrame()

if not os.path.exists("ImagemClassificar"):
    os.makedirs("ImagemClassificar")
else:
    shutil.rmtree("ImagemClassificar")
    os.makedirs("ImagemClassificar")

for cnt in contornos:
    #area
    area = cv2.contourArea(cnt)
        
    #perimetro
    perimetro = cv2.arcLength(cnt,True)

    #compacidade
    compacidade = (perimetro**2)/area
    #print("Compacidade: ", compacidade)

    #circulatidade
    circularidade = (4*np.pi*area)/perimetro**2
    #print("circulatidade: ", circulatidade)

    #Cria eclipse envolvente 
    (xc,yc), (d1,d2), angle  = cv2.fitEllipse(cnt)

    #Pegas os pontos do eixo maior
    rmajor = max(d1,d2)/2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90

    x1 = xc + math.cos(math.radians(angle))*rmajor
    y1 = yc + math.sin(math.radians(angle))*rmajor
    x2 = xc + math.cos(math.radians(angle+180))*rmajor
    y2 = yc + math.sin(math.radians(angle+180))*rmajor
    #calcula distancia
    eixo_maior = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    #Pegas os pontos do eixo menor
    rminor = min(d1,d2)/2
    if angle > 90:
        angle = angle - 90
    else:
        angle = angle + 90

    x1 = xc + math.cos(math.radians(angle))*rminor
    y1 = yc + math.sin(math.radians(angle))*rminor
    x2 = xc + math.cos(math.radians(angle+180))*rminor
    y2 = yc + math.sin(math.radians(angle+180))*rminor
    #calcula distancia
    eixo_menor = np.sqrt((x2-x1)**2 + (y2-y1)**2)

    #Razão do eixo menor pelo eixo maior
    razao_eixos = eixo_menor/eixo_maior
    #print("Razão Eixos: ", razao_eixos)

    
    mask = np.zeros(imagem_original.shape,np.uint8)
    mask = cv2.drawContours(mask,[cnt],0,255,-1)
    #pixelpoints = np.transpose(np.nonzero(mask))
    
    mean_val = cv2.mean(imagem_original,mask = mask)
    auxiliar = mean_val[0]/255

    df = df.append({'Compacidade': compacidade, 'Circularidade': circularidade, 'Razão Eixos': razao_eixos,  'Intencidade Média': auxiliar}, ignore_index=True)

    imagem_contornos = cv2.drawContours(cv2.cvtColor(imagem_original, cv2.COLOR_BAYER_BG2BGRA), [cnt], 0, (222,0,0), 3)

    cv2.imwrite('ImagemClassificar\\' +str(aux) + ".bmp", imagem_contornos)
    #print("Intencidade media : ", razao_eixos)
    #print("")

    aux += 1

df.to_csv('teste.csv', index=False)

# Carregar modelo
with open('MLP\Classificador.pkl', 'rb') as f:
    model = pickle.load(f)

data = pd.read_csv('teste.csv')

X_normalized = normalize(data, axis=0)

sc=StandardScaler()
scaler = sc.fit(data)

testX_scaled = scaler.transform(data)

y_pred = model.predict(testX_scaled)


F = 0
A = 0
Q = 0
P = 0
C = 0
U = 0

for x in y_pred:
    if x == 0:
        F += 1
    if x == 1:
        A += 1
    if x == 2:
        Q += 1
    if x == 3:
        P += 1
    if x == 4:
        C += 1
    if x == 5:
        U += 1

os.system('cls' if os.name == 'nt' else 'clear')
#print(y_pred)
print("Arquivo: ", nome_arquivo)
print("Numero de folhas: ", size)

if A > 0:
    print("Araçá (", A,")")
if C > 0:
    print("Coleus (", C,")")
if F > 0:
    print("Folhado (", F,")")
if P > 0:
    print("Pessegueiro (", P,")")
if Q > 0:
    print("Quaresmeira (", Q,")")
if U > 0:
    print("Uva do mato (", U,")")

print("")