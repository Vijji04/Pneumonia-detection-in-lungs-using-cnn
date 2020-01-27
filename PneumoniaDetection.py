#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
print(os.listdir("../input"))
import numpy as np
import matplotlib.pyplot as plt
from glob import glob
from keras.preprocessing.image import ImageDataGenerator


# In[ ]:


path = "../input/chest_xray/chest_xray"
dirs  = os.listdir(path)
print(dirs)


# In[ ]:


train_folder = path + '/train/'
test_folder  = path + '/test/'
val_folder   = path + '/val/'

train_dirs = os.listdir(train_folder)
print(train_dirs)


# In[ ]:


train_normal = train_folder + 'NORMAL/'
train_pneu   = train_folder + 'PNEUMONIA/'


# In[ ]:


pneu_images   = glob(train_pneu + "*.jpeg")
normal_images = glob(train_normal + "*.jpeg")


# In[ ]:


def show_imgs(num_of_imgs):
    
    for img in range(num_of_imgs):
        pneu_pic   = np.asarray(plt.imread(pneu_images[img]))
        normal_pic = np.asarray(plt.imread(normal_images[img]))

        fig = plt.figure(figsize= (15,10))

        normal_plot = fig.add_subplot(1,2,1)
        plt.imshow(normal_pic, cmap='gray')
        normal_plot.set_title('Normal')
        plt.axis('off')

        pneu_plot = fig.add_subplot(1, 2, 2)
        plt.imshow(pneu_pic, cmap='gray')
        pneu_plot.set_title('Pneumonia')
        plt.axis('off')
    
        plt.show()


# In[ ]:


train_datagen = ImageDataGenerator(rescale            = 1/255,
                                   shear_range        = 0.2,
                                   zoom_range         = 0.2,
                                   horizontal_flip    = True,
                                   rotation_range     = 40,
                                   width_shift_range  = 0.2,
                                   height_shift_range = 0.2)


# In[ ]:


test_datagen = ImageDataGenerator(rescale = 1/255)


# In[ ]:


training_set = train_datagen.flow_from_directory(train_folder,
                                   target_size= (64, 64),
                                   batch_size = 32,
                                   class_mode = 'binary')

val_set = test_datagen.flow_from_directory(val_folder,
                                   target_size=(64, 64),
                                   batch_size = 32,
                                   class_mode ='binary')

test_set = test_datagen.flow_from_directory(test_folder,
                                   target_size= (64, 64),
                                   batch_size = 32,
                                   class_mode = 'binary')


# In[ ]:


from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Dropout


# In[ ]:


model = Sequential()

model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3), padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation = 'relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))


# In[ ]:


model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])


# In[ ]:


model_train = model.fit_generator(training_set,
                         steps_per_epoch = 200,
                         epochs = 5,
                         validation_data = val_set,
                         validation_steps = 100)


# In[ ]:


test_accuracy = model.evaluate_generator(test_set,steps=624)

print('Testing Accuracy: {:.2f}%'.format(test_accuracy[1] * 100))

