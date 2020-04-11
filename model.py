import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd
import os
from PIL import Image
import tensorflow as tf

resize_method = Image.ANTIALIAS

#season_path=r"C:\Users\Administrator\Desktop\season"

saved_path=r"C:\Users\Administrator\Desktop\AI-data\Dataset"

set_width=120
set_height=120

#Resize image
extensions= ['PNG','JPG','JPEG','JFIF']

def adjusted_size(width,height):
    return width,height

def converting(img_num,dir):	
#def converting(dir):	    
    count=img_num
   
    if __name__ == "__main__":
        for f in os.listdir(dir):
            if os.path.isfile(os.path.join(dir,f)):
                f_text, f_ext= os.path.splitext(f)
                f_ext= f_ext[1:].upper()
                if f_ext in extensions:             
                    count+=1                    
                    #print(f_ext)
                    #print(f)
                    image = Image.open(os.path.join(dir,f))                               
                    image = image.resize(adjusted_size(set_width, set_height))
                    image.save(os.path.join(saved_path,str(count)+'.png'))
                    
"""  
converting(0,path_foto)                    
converting(foto_len,path_scan)
"""
#converting(season_path)


dir_temp=['altar','apse','bell_tower','column',
             'dome(inner)','dome(outer)','flying_buttress',
             'gargoyle','stained_glass','vault'
             ]
counts=0
for place in dir_temp:
    arc_path=r"C:\Users\Administrator\Desktop\architecture\\"+place
    print(arc_path)    
    print(counts)
    converting(counts,arc_path)
    counts+=len(os.listdir(arc_path))




"""Data Preprocessing"""
#Make a list of image data
len(os.listdir(saved_path))
getImg = sorted(os.listdir(saved_path),key=len)
immatrix = np.array(
        [np.array(Image.open(saved_path + '\\' + img_name).convert(mode='L')).flatten() for img_name in getImg],
        order='f'
        )

# 0-> foto, 1-> scan
label = np.empty([len(getImg),],dtype = int)
#label[0:65]=0
#label[65:149]=1


#Architecture
label[0:829] = 0
label[829:1343] = 1
label[1343:2402] = 2
label[2402:4321] = 3
label[4321:4937] = 4
label[4937:6114] = 5
label[6114:6521] = 6
label[6521:8092] = 7
label[8092:9125] = 8
label[9125:10235] = 9


"""
Season
label[0:300] = 0 #Cloudy
label[300:514] = 1 #Rain
label[514:765] = 2 #Shine
label[765:1121] = 3 #Sunrise
"""


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
y_train=y_train[:,0:4]
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
model.add(Dense(10, activation='softmax'))    


model.compile(loss='categorical_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy'])

history = model.fit(x_train, y_train,
          batch_size=200,
          epochs=20,
          verbose=1,
          validation_data=(x_test, y_test))

loss, acc = model.evaluate(x_test, y_test)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

y_pred=model.predict(x_test)


"""
from sklearn.metrics import confusion_matrix as cm
matrix=cm(y_test,y_pred)

import keras

model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(set_width,set_height,1)))
model.add(BatchNormalization())

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())

model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(4, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adam(),
              metrics=['accuracy'])

y_pred=model.predict([x_test])
x_test=list(x_test)
"""