import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os
from PIL import Image
import tensorflow as tf


resize_method = Image.ANTIALIAS

path_foto= r"C:\Users\Administrator\Desktop\AI-data\Kartu\Foto" #Ganti ama directory folder data foto
path_scan= r"C:\Users\Administrator\Desktop\AI-data\Kartu\Scan" #Ganti ama directory folder data scan

saved_path=r"C:\Users\Administrator\Desktop\AI-data\Dataset" #directory lu taroh hasil outputnya 

foto_len= len(os.listdir(path_foto))
scan_len= len(os.listdir(path_scan))

set_width=300
set_height=300


#Resize image
extensions= ['PNG','JPG','JPEG']

def adjusted_size(width,height):
    return width,height

def converting(img_num,dir):	
    count=img_num
    if __name__ == "__main__":
        for f in os.listdir(dir):
            if os.path.isfile(os.path.join(dir,f)):
                f_text, f_ext= os.path.splitext(f)
                f_ext= f_ext[1:].upper()
                if f_ext in extensions:             
                    count+=1
                    image = Image.open(os.path.join(dir,f))                               
                    image = image.resize(adjusted_size(set_width, set_height))
                    image.save(os.path.join(saved_path,str(count)+".png"))
                    
converting(0,path_foto)                    
converting(foto_len,path_scan)


"""Data Preprocessing"""
#Make a list of image data
getImg = sorted(os.listdir(saved_path),key=len)
immatrix = np.array(
        [np.array(Image.open(saved_path + '\\' + img_name).convert(mode='L')).flatten() for img_name in getImg],
        order='f'
        )

# 0-> foto, 1-> scan
label = np.empty([len(getImg),],dtype = int)
label[0:33] = 0
label[33:64] = 1

from sklearn.utils import shuffle
data,label = shuffle(immatrix,label, random_state = 99)
train_data = [data,label]

from sklearn.model_selection import train_test_split as tts
x_train,x_test,y_train,y_test=tts(train_data[0],train_data[1],test_size=0.2,random_state=0)


#Normalize train data
x_train = x_train.reshape(x_train.shape[0], set_width, set_height, 1)
x_train = x_train.astype('float32')
x_train /= 255

x_test = x_test.reshape(x_test.shape[0], set_width, set_height, 1)
x_test = x_test.astype('float32')
x_test /= 255


from keras.utils import to_categorical as tc
y_train=tc(y_train)
y_test=tc(y_test)

plt.imshow(x_test[10].reshape(set_width,set_height), interpolation = 'nearest')



"""Making model or classifier"""
from keras.models import Sequential 
from keras.layers import BatchNormalization
from keras.layers import Conv2D, MaxPooling2D 
from keras.layers import Dense, Dropout 
from keras.layers import Flatten
from keras.utils import np_utils
from keras.optimizers import SGD

model = Sequential()
model.add(BatchNormalization(input_shape=(set_width,set_height,1)))
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2,2)))
model.add(Dropout(0.25))
    
model.add(BatchNormalization())
model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))    


model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=5,
          epochs=100,
          verbose=1,
          validation_data=(x_test, y_test))