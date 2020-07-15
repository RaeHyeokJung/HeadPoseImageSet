!wget http://www-prima.inrialpes.fr/perso/Gourier/Faces/HeadPoseImageDatabase.tar.gz
!tar -xvzf HeadPoseImageDatabase.tar.gz


import numpy as np
import matplotlib.pyplot as plt
import glob
import math
import cv2
from tensorflow.keras import layers
from tensorflow.keras import models
from tensorflow.keras import optimizers
import os


# ********************************************************
#               HeadPoseImageDatabase
# ********************************************************

train_person_list = [1,2,3,4,5,6,7,8,9,10,11,12,13,14]
test_person = [15]

person_list = sorted(glob.glob('Person*')) # search the folders starting with 'Person'

test_img_list = []
test_img_list.extend(glob.glob(os.path.join(person_list[14], '*.jpg')))
test_idx = [i for i, f in enumerate(test_img_list)]


for train_person in train_person_list :
        

    img_list = []
    img_list.extend(glob.glob(os.path.join(person_list[train_person], '*.jpg')))

    N = len(img_list)  # total number of images
    X = np.zeros((N, 288, 384, 3))  # {numpy} -> shape : [2790, 288, 384, 3]
    Y = np.zeros((N, 4))  # {numpy} -> shape : [2790, 4]

    for i, img_file in enumerate(img_list):
        img = cv2.imread(img_file, cv2.IMREAD_COLOR)  # read the image file
        txt_file = img_file[:-4] + '.txt'  # take the txt file address from the image file address
        with open(txt_file, 'r') as f:
            line = f.read().splitlines()
            center_x = int(line[3])
            center_y = int(line[4])
            width = int(line[5])
            height = int(line[6])  # open the text file and take the face box information from it

        X[i] = img  # put the image into the array 'X'
        Y[i] = center_x, center_y, width, height  # put the box info into the array'Y'

    train_idx = [i for i, f in enumerate(img_list)]
    


    X_train = X[train_idx,:,:,:]
    Y_train = Y[train_idx,:]
    X_test = X[test_idx,:,:,:]
    Y_test= Y[test_idx,:]
    del X
    del Y

    # Convert class vectors to binary class matrices.
    (train_num, h, w, channum) = X_train.shape
    (test_num, _, _, _) = X_test.shape
    (_, outputdim) = Y_train.shape
    # ********************************************************
    #               Training
    # ********************************************************

    # Training Parameters
    epochs = 50
    batch_size = 32

    x_train = X_train.astype('float32')
    x_test = X_test.astype('float32')
    x_train /= 255
    x_test /= 255
    _,h,w,channum = x_train.shape

    mean = Y_train.mean(axis=0)
    std = Y_train.std(axis=0)
    y_train = (Y_train - mean) / std
    y_test = (Y_test - mean) / std




    # *************************************************************
    #               Model building
    # *************************************************************
    print("Model building start")
    if train_person == 1 :
        model = models.Sequential()
        model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu', input_shape=(h, w, channum)))
        model.add(layers.Conv2D(16, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(layers.Conv2D(32, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='same', activation='relu'))
        model.add(layers.Conv2D(64, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='relu'))
        model.add(layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
        model.add(layers.Flatten())
        model.add(layers.Dense(10000, activation='relu'))
        model.add(layers.Dropout(0.2))
        model.add(layers.Dense(4, activation='linear'))
        model.summary()

        model.compile(optimizer=optimizers.Adam(lr=0.001),
                      loss='mean_squared_error',
                      metrics=['mse','mae'])
        print("model compile")

    history = model.fit(x=x_train, y=y_train, epochs=epochs, batch_size=batch_size, validation_data=(x_test, y_test),
                        verbose=1, shuffle=True)
    print("model fit")


# *************************************************************
#               Visualization
# *************************************************************

# Training loss
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(loss) + 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()



# Training accuracy
plt.clf()   # 그래프를 초기화합니다
acc = history.history['mae']
val_acc = history.history['val_mae']
plt.plot(epochs, acc, 'bo', label='Training mean_absolute_error')
plt.plot(epochs, val_acc, 'b', label='Validation mean_absolute_error')
plt.title('Training and validation mean_absolute_error')
plt.xlabel('Epochs')
plt.ylabel('mean_absolute_error')
plt.legend()
plt.show()


# Training accuracy
plt.clf()   # 그래프를 초기화합니다
acc = history.history['mse']
val_acc = history.history['val_mse']
plt.plot(epochs, acc, 'bo', label='Training mean_squared_error')
plt.plot(epochs, val_acc, 'b', label='Validation mean_squared_error')
plt.title('Training and validation mean_squared_error')
plt.xlabel('Epochs')
plt.ylabel('mean_squared_error')
plt.legend()
plt.show()



# Apply NN to training and test set
scores = model.evaluate(x_train, y_train, verbose=2)
yhat_train = model.predict(x_train)
error = y_train - yhat_train
print(error)


# ********************************************************
#  Prediction visulaization
# ********************************************************

# Test data
yhat_test = model.predict(x_test)
print('MSE for test set = {:f}'.format(
    np.trace(np.matmul((y_test-yhat_test).T,(y_test-yhat_test)))/(outputdim*test_num)))
print('MAE for test set = {:f}'.format(
    np.sum(np.abs(y_test-yhat_test))/(outputdim*test_num)))
Yhat_test = yhat_test*std+mean


fig = plt.figure()
ims = np.random.randint(test_num, size=4)
for i in range(4):
    subplot = fig.add_subplot(2,2, i+1)
    sample_idx = ims[i]
    sample = X_test[sample_idx].copy()  # copy the image to sample ( not to affect the original data )

    sample_cx = Y_test[sample_idx][0]
    sample_cy = Y_test[sample_idx][1]
    sample_w = Y_test[sample_idx][2]
    sample_h = Y_test[sample_idx][3] # get box info from Y

    sample_x1 = int(sample_cx - sample_w//2)
    sample_y1 = int(sample_cy - sample_h//2)
    sample_x2 = int(sample_cx + sample_w//2)
    sample_y2 = int(sample_cy + sample_h//2)
    cv2.rectangle(sample, (sample_x1, sample_y1), (sample_x2, sample_y2), (255, 0, 0), (3))


    sample_cx = Yhat_test[sample_idx][0]
    sample_cy = Yhat_test[sample_idx][1]
    sample_w = Yhat_test[sample_idx][2]
    sample_h = Yhat_test[sample_idx][3] # get box info from Y

    sample_x1 = int(sample_cx - sample_w//2)
    sample_y1 = int(sample_cy - sample_h//2)
    sample_x2 = int(sample_cx + sample_w//2)
    sample_y2 = int(sample_cy + sample_h//2)
    cv2.rectangle(sample, (sample_x1, sample_y1), (sample_x2, sample_y2), (0, 0, 255), (3))

    subplot.set_xticks([])
    subplot.set_yticks([])
    subplot.imshow(cv2.cvtColor(sample.astype('uint8'), cv2.COLOR_BGR2RGB))

plt.show()