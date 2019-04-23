
from keras.datasets import mnist
from keras.models import load_model
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
###################################################################### Load model
model = load_model('model.h5')
print('load ok')
###################################################################### Load data
#--------------------------data from MMIST
#(X_train, y_train), (X_test, y_test) = mnist.load_data()
#test = X_test[0].reshape(1,28,28,1)
# Show
# print(X_test.shape)
# plt.imshow(X_test[0])
# plt.show()
#--------------------------------------------------C2
###################################################################### data tự chuẩn bị
# Mode "L" :greyscale (8-bit pixels, black and white)
img = Image.open('cat_or_dog_1.jpg').convert('RGB')
img = img.resize((64,64))
imgArr = np.array(img)
print(imgArr.shape)
imgArr = imgArr.reshape(1,imgArr.shape[0],imgArr.shape[1],3)# shape that CNN expects is a 4D array (batch, height, width, channels) <batch: số samples>
#print(imgArr)
plt.imshow(img)
plt.show()

####################################################################### Predicting the Test set results
y_pred = model.predict(imgArr)
print("Predict results: \n", y_pred[0])
print("Predict label: \n", np.argmax(y_pred[0]))
# print(y_pred)

