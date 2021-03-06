import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from trian2 import modelo
from tensorflow.keras.models import load_model

longitud, altura = 150, 150
pesos_modelo = './first_try.h5'
cnn = modelo()
cnn.load_weights(pesos_modelo)

def predict(file):
  x = load_img(file, target_size=(longitud, altura))
  x = img_to_array(x)
  x = np.expand_dims(x, axis=0)
  array = cnn.predict(x)
  result = array[0]
  answer = np.argmax(result)
  if answer == 0:
    print("Ambulancia")
  elif answer == 1:
    print("Bicicleta")
  elif answer == 2:
    print("Bus")
  elif answer == 3 :
    print("Camion")
  elif answer == 4 :
    print("Camioneta")
  elif answer == 5 :
    print("Coche")
  elif answer == 6 :
    print("Excavadora")
  elif answer == 7 :
    print("Limosina")
  elif answer == 8 :
    print("Motocarro")
  elif answer == 9 :
    print("Motocicleta")

  return answer

for x in range(1,15):
  y = str(x)
  predict('./predecir/prueba ('+y+').jpg')
  print(y)
  print()

'''predict('./Definir/ambulancia (1).jpg')
predict('./Definir/ambulancia (2).jpg')
predict('./Definir/bicicleta (1).jpg')
predict('./Definir/bicicleta (2).jpg')
predict('./Definir/bus (1).jpg')
predict('./Definir/bus (2).jpg')
predict('./Definir/Camion (1).jpg')
predict('./Definir/Camion (2).jpg')
predict('./Definir/camioneta (1).jpg')
predict('./Definir/camioneta (2).jpg')
predict('./Definir/coche (1).jpg')
predict('./Definir/coche (2).jpg')
predict('./Definir/excavadora (1).jpg')
predict('./Definir/excavadora (1).jpg')
predict('./Definir/lumusina (1).jpg')
predict('./Definir/lumusina (2).jpg')
predict('./Definir/motocarro (1).jpg')
predict('./Definir/motocarro (2).jpg')
predict('./Definir/motocicleta (1).jpg')
predict('./Definir/motocicleta (2).jpg')'''
