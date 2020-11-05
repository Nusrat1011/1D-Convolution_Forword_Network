#!/usr/bin/env python
# coding: utf-8

# In[37]:


import numpy as np
import keras
from keras.layers import Conv1D, Dense, Dropout, MaxPooling1D, Flatten,GlobalMaxPooling1D
from keras.models import Sequential
from keras import optimizers
from keras.optimizers import Adam 
from sklearn.model_selection import train_test_split
from sklearn import metrics
import matplotlib.pyplot as plt


# In[38]:


class LossHistory(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.losses = {'batch':[], 'epoch':[]}
        self.accuracy = {'batch':[], 'epoch':[]}
        self.val_loss = {'batch':[], 'epoch':[]}
        self.val_acc = {'batch':[], 'epoch':[]}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.accuracy['batch'].append(logs.get('acc'))
        self.val_loss['batch'].append(logs.get('val_loss'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.losses['epoch'].append(logs.get('loss'))
        self.accuracy['epoch'].append(logs.get('acc'))
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.losses[loss_type]))
        plt.figure()
        plt.plot(iters, self.losses[loss_type], 'g', label='train loss')
        plt.grid(True)
        plt.xlabel(loss_type)
        plt.ylabel('train-loss')
        plt.legend(loc="upper right")
        plt.show()


# In[39]:


output = np.loadtxt('Absorption_40.txt')
input = np.loadtxt('Thickness_40.txt')

output= np.array(output)
input= np.array(input)
train_input, val_input, train_output, val_output = train_test_split(input, output, test_size=0.2, shuffle= True)
print(train_input.shape)
print(train_output.shape)


# In[40]:


model = Sequential()

train_input = train_input.reshape(train_input.shape[0], 9, 1).astype('float32')
val_input = val_input.reshape(val_input.shape[0], 9, 1).astype('float32') 

model = Sequential()
model.add(Conv1D(filters=32, kernel_size=3, input_shape=(9, 1)))
model.add(MaxPooling1D(pool_size=3 ))
model.add(Dropout(0.1))
model.add(Flatten())
model.add(Dense(200, activation='relu'))
model.summary()


# In[41]:


adam = optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-07, amsgrad=False)

model.compile(loss='mse',optimizer='adam' ,metrics=['acc'])


# In[42]:


history = LossHistory()

model.fit(train_input, train_output, batch_size=128, epochs=100, shuffle=True,verbose=2,validation_data=(val_input, val_output), callbacks=[history])


# In[43]:


history.loss_plot('epoch')


# In[44]:


predicted_output = model.predict(val_input) # get prediction for test_input
errors = list()
for true_val, pred_val in zip(val_output, predicted_output):
    temp_error = metrics.mean_absolute_error(true_val, pred_val) 
    errors.append(temp_error)
errors = np.asarray(errors)


# In[45]:


plt.figure()
x = range(len(errors))
plt.plot(x, errors)
plt.xlabel('Test Samples')
plt.ylabel('Prediction Error')
plt.show()


# In[46]:


structure=(1,45,4.2,65,10,6.6,4.4,10,6.6) # Proposed Multilayer Structure Layer Thicknesess
structure=np.array(structure).reshape(-1,9,1)
spectrum_predict=model.predict(structure)
spectrum_predict=np.array(spectrum_predict).reshape(200)


# In[48]:


with open('Wavelength_40.txt') as f:
    lines = f.readlines()
    x1 = [line.split()[0] for line in lines]

for i in range(0, len(x1)): 
    x1[i] = float(x1[i]) 

x1 = np.reshape(x1,(200,1)) 
x1 = x1.flatten() 

with open('Wavelength_300_25000_Aem_11.txt') as f:
    lines = f.readlines()
    x2 = [line.split()[0] for line in lines]

for i in range(0, len(x2)): 
    x2[i] = float(x2[i]) 

x2 = np.reshape(x2,(200,1)) 
x2 = x2.flatten() 
with open('Emissivity_300_25000_Aem_11.txt') as f:
    lines = f.readlines()
    y2 = [line.split()[0] for line in lines]

for i in range(0, len(y2)): 
    y2[i] = float(y2[i]) 

y2 = np.reshape(y2,(200,1)) 
y2 = y2.flatten() 

f = plt.figure(figsize=(5,5))
ax = f.add_subplot(211)
ax.plot(x1, abs(spectrum_predict), lw=2, color='blue',label='Predicted')
ax.set_xlabel('Wavelength (um)')
ax.set_ylabel('Emissivity')
ax.grid(True)

ax.plot(x2, y2, lw=2, label='Real', color='green')
ax.set_xlabel('Wavelength (um)')
ax.set_ylabel('Emissivity')
ax.grid(True)
#plt.figure()
#plt.plot(x1, abs(spectrum_predict)) # Absorption Spectrum
#plt.grid(True)
#plt.xlabel('wavelength in um')
#plt.ylabel('spectrum_predict')
#plt.show()
# plt.savefig('Spectrum_predicted_vs_wavelength.jpg')
# plt.close()


# In[49]:


f=open('Predicted_revised.txt','a')
for i in range(200):
    for j in range(1):
        f.write(str(spec_prec[i,j]))
    f.write('  ')

f.close()


# In[ ]:




