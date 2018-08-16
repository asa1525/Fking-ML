
# coding: utf-8
## kears mnist
import keras
from keras.models import Sequential
from keras.layers import Dense,Dropout,Flatten,MaxPool2D,Conv2D
import pandas as pd
import numpy as np
from keras import backend as K


BATCH_SIZE = 128
NUM_CLASSES = 10
EPOCHS = 12
img_rows ,img_cols = 28,28


train_data = pd.read_csv('datasets/digit/train.csv')
test_data = pd.read_csv('datasets/digit/test.csv')
train_x ,train_y = train_data.values[:,1:] , train_data.values[:,0]
test_x = test_data.values
## 将 train_y 和test_y 进行都热向量编码
train_y = keras.utils.np_utils.to_categorical(train_y,num_classes=NUM_CLASSES)
## 将 traion_x 和 test_x 进行reshape


train_x = np.reshape(train_x,[-1, 28, 28, 1])
print(np.shape(train_x))
test_x = np.reshape(test_x,[-1, 28, 28, 1])
print(np.shape(test_x))

print(K.image_data_format())


## 归一化
train_x = train_x.astype('float32') / 255
test_x = test_x.astype('float32') / 255
input_shape = (img_rows, img_cols, 1)

## 构造网络模型
model = keras.models.Sequential()


# 增加 通道为32 的 卷积核大小为3 的卷积  激活函数为relu  输入图片的大小为 28 * 28 * 1
model.add(Conv2D(32,(3,3),
                 activation='relu',
                 input_shape=(28,28,1)
                ))


model.add(Conv2D(64,(3,3),
                 activation='relu',
                 input_shape=(28,28,1)
                ))
## 池化层
model.add(MaxPool2D(pool_size=(2,2)))
# 增加dropout层
model.add(Dropout(0.5))

## 增加flattern层 把结果拉成一个列向量
model.add(Flatten())
## 加一个全连接层
model.add(Dense(128,activation='relu'))
model.add(Dropout(0.5))
## 再加一个全链接层  softmax来预测全概率
model.add(Dense(NUM_CLASSES,activation='softmax'))

## 编译模型
model.compile(loss=keras.metrics.categorical_crossentropy,optimizer=keras.optimizers.Adam(),metrics=['accuracy'])


model.fit(train_x, train_y, batch_size=BATCH_SIZE, epochs=EPOCHS,
          verbose=1)

CLASSES = model.predict_classes(test_x,batch_size=BATCH_SIZE)
pd.DataFrame(
    {"ImageId": range(1, len(CLASSES) + 1), "Label": CLASSES}
).to_csv('output.csv', index=False, header=True)
print('done.')
