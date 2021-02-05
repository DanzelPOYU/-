

import keras
keras.__version__
from keras.datasets import mnist
import numpy as np
from keras import models
from keras import layers
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28 * 28))
#將二為矩陣轉為一為向量供類神經模組讀取
train_images = train_images.astype('float32') / 255
#PIXEL標準化，閥值(0,1)供電腦判斷
test_images = test_images.reshape((10000, 28 * 28))
#將二為矩陣轉為一為向量供類神經模組讀取
test_images = test_images.astype('float32') / 255
#PIXEL標準化，閥值(0,1)供電腦判斷
train_labels_categ = to_categorical(train_labels)
test_labels_categ = to_categorical(test_labels)
#訓練與測試輸出進行分類
#此題為例，把輸出0分類為[1,0,0,0,0,0,0,0,0,0]
        #把1分類為[0,1,0,0,0,0,0,0,0,0]
        #依此類推

network = models.Sequential()
network.add(layers.Dense(512, activation='relu'))#, input_shape=(28 * 28,)))
#建立神經元有512個的類神經層，activation='relu'激率(刺激率)
network.add(layers.Dense(10, activation='softmax'))
#建立神經元有10個的類神經層，activation='softmax'分類
network.compile(optimizer='rmsprop',#編譯模組，定義優化器
                loss='categorical_crossentropy',#分類
                metrics=['accuracy'])#準確率


network.fit(train_images, train_labels_categ, epochs=5, batch_size=128)
#進行訓練輪迴5次，每次訓練為128，總訓練次數為5*(60000/128)

test_loss, test_acc = network.evaluate(test_images, test_labels_categ)
#預測結果精準率
print('test_acc:', test_acc)
print('test_loss:', test_loss)