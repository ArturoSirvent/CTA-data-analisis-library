import numpy as np
import os 
import tensorflow as tf
import glob


#un modelo que nos crea un classificador con parametros
def create_model(input_shape=(55,93,1),filts=None,kerns=None):
    input=tf.keras.Input(shape=input_shape)
    if filts is None:
        filts=[[32,64],[128,256],[64,12],[12,4]]
    if kerns is None:
        kerns=[[3,3],[3,3],[3,3],[3,3]]

    for i in range(len(filts)):
        if i==0:
            x=tf.keras.layers.Conv2D(filts[i][0],kerns[i][0],activation="relu",padding="same")(input)
        else:
            x=tf.keras.layers.Conv2D(filts[i][0],kerns[i][0],activation="relu",padding="same")(x)

        x=tf.keras.layers.Conv2D(filts[i][1],kerns[i][1],activation="relu",padding="same")(x)

        x=tf.keras.layers.MaxPool2D(pool_size=(2,2))(x)
    x=tf.keras.layers.Flatten()(x)
    x=tf.keras.layers.Dense(100,activation="relu")(x)
    #model.add(tf.keras.layers.Dense(20,activation="relu"))
    final=tf.keras.layers.Dense(3,activation="softmax")(x)
    model=tf.keras.Model(inputs=input,outputs=final)
    return model


#?? it would be cool to have a function that computes the graph for the best LR for our model
#vamos a ver como va el learning rate, lo actualizamos exponencialmente
def funcion_actualizacion(epoch,lr):
    #queremos que empiece en 10^-10 y vaya hasta 10^-1 
    return lr*10**(np.log(10**7)/100)

#this function is needed, but also the model and some more things...
"""
actualizar=tf.keras.callbacks.LearningRateScheduler(funcion_actualizacion)
model1.compile(optimizer=tf.keras.optimizers.Adam(1e-7),loss='mse')
aux2=model1.fit(x_train,y_train,epochs=100,validation_data=(x_test,y_test),callbacks=[actualizar]) 
lr_0=1e-7
epochs=100
lrates=[lr_0*10**(np.log10(10**6)*i/epochs) for i in range(1,epochs+1) ]
plt.figure(figsize=(13,6))
plt.plot(lrates,aux2.history["val_loss"])
plt.plot(lrates,aux2.history["loss"])

plt.xscale("log")
plt.xlabel("Epoch",fontsize=14)
plt.ylabel("Loss function",fontsize=14)
plt.grid(alpha=0.5)
plt.tight_layout()
"""


def model_1_tel(input_shape=(55,93,1),filtros=None,batch_init=True,last_layers=None,avg_pooling=False,classes=3,learning_rate=1e-5,first_model=None,first_part=False):
    if filtros is None:
        filtros=[[64,32],[128,64,64],[32,16]]
    if last_layers is None:
        last_layers=[35,20]
    #lo vamos a hacer super customizable para probar esa vaina de entrenar muchos hiperparametros
    #estructura lo que nos mete son el numero de filtros conv y maxpool
    model=tf.keras.Sequential([tf.keras.layers.InputLayer(input_shape=input_shape)])
    if first_model:
        first_model.trainable=False
        model.add(first_model)
    if batch_init:
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Activation("relu"))
    for i,j in enumerate(filtros):
        #aqui se alterna entre conv y maxpool(1) o average pool(-1)
        for k in j:
            model.add(tf.keras.layers.Conv2D(k,3,activation="relu",padding="same"))
        if avg_pooling:
            model.add(tf.keras.layers.AveragePooling2D((2,2)))
        else:
            model.add(tf.keras.layers.MaxPool2D((2,2)))
    model.add(tf.keras.layers.Flatten())
    if first_part:
        return model
    for j,i in enumerate(last_layers):
        model.add(tf.keras.layers.Dense(i,activation="relu"))
    model.add(tf.keras.layers.Dense(classes,activation="softmax"))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss="categorical_crossentropy",metrics=["accuracy"])

    return model



#??this functions has a kernel_regularizer, is it always nice to have it?
def model_multi_tel(len_inputs=4,input_shapes=[(55,93,1)],classes=7,learning_rate=1e-5,pre_proces_model=None,filtros=None,last_dense=None,autoencoder=None,common_pre=True,batch=False):
    #this function creates a cnn model
    inputs=[]
    outputs=[[] for i in range(len_inputs)]
    if len(input_shapes)==1:
        for i in range(1,len_inputs):
            input_shapes.append(input_shapes[0])

    for i in range(len_inputs):
        inputs.append(tf.keras.Input(shape=input_shapes[i]))

    if pre_proces_model:
        for i in range(len_inputs):
            outputs[i]=pre_proces_model(first_model=autoencoder)(inputs[i])
    else:
        if filtros is None:
            filtros=[[64,128],[128,254,64],[32]]
        #si no le metemos un modelo pues habra que meter aqui chicha porque sino se queda esto muy vacio
        if common_pre:
            pre_model=model_1_tel(input_shapes[0],filtros=filtros,first_part=True,first_model=autoencoder,batch_init=batch)
            outputs[0]=pre_model(inputs[0])
            for i in range(1,len_inputs):
                if input_shapes[i]!=input_shapes[i-1]:
                #esto esta suponiendo que ponemos juntos los que tienen igual shape
                    pre_model=model_1_tel(input_shapes[i],filtros=filtros,first_part=True,first_model=autoencoder,batch_init=batch) 
                outputs[i]=pre_model(inputs[i])

        else:
            for i in range(len_inputs):
                #SI QUEREMOS PONER AUTOENCODER TENEMOS QUE VER LA FOTMA DE COPIARLO
                pre_model=model_1_tel(input_shape=input_shapes[i],filtros=filtros,first_part=True,batch_init=batch)
                outputs[i]=pre_model(inputs[i]) 
    #nos falta la ultima parte
    if last_dense is None:
        last_dense=[65,35]
    x=tf.keras.layers.concatenate(outputs)
    for i in last_dense:
        #?? we add a kernel_regularizer for some reason so, it would be great to know if it does always help
        x=tf.keras.layers.Dense(i,activation="relu",kernel_regularizer="l2")(x)
    end_layer=tf.keras.layers.Dense(classes,activation="softmax")(x)
    model=tf.keras.Model(inputs=inputs,outputs=end_layer)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss="categorical_crossentropy",metrics=["accuracy"])

    return model


#?? FILTROS IS NOT SPECIFIED IN THE ARGUMENTS!! SHOULD I?
def model_multi_tel_encoder(len_inputs=4,input_shapes=[(55,93,1)],classes=3,last_dense=None,encoder=None,common_pre=True): #,learning_rate=1e-5
    inputs=[]
    outputs=[[] for i in range(len_inputs)]
    if len(input_shapes)==1:
        for i in range(1,len_inputs):
            input_shapes.append(input_shapes[0])

    for i in range(len_inputs):
        inputs.append(tf.keras.Input(shape=input_shapes[i]))


    if common_pre:
        outputs[0]=encoder(inputs[0])
        for i in range(1,len_inputs):
            #if input_shapes[i]!=input_shapes[i-1]:
            #esto esta suponiendo que ponemos juntos los que tienen igual shape
            #    pre_model=model_1_tel(input_shapes[i],filtros=filtros,first_part=True,first_model=autoencoder) 
            outputs[i]=encoder(inputs[i])

    else:
        for i in range(len_inputs):
            #SI QUEREMOS PONER AUTOENCODER TENEMOS QUE VER LA FOTMA DE COPIARLO
            #?? FILTROS IS NOT SPECIFIED IN THE ARGUMENTS!! SHOULD I?
            pre_model=model_1_tel(input_shape=input_shapes[i],filtros=filtros,first_part=True,batch_init=batch)
            outputs[i]=pre_model(inputs[i]) 
    #nos falta la ultima parte
    if last_dense is None:
        last_dense=[65,35]
    x=tf.keras.layers.concatenate(outputs)
    for i in last_dense:
        #??kernel regularizer, should it be always there?
        x=tf.keras.layers.Dense(i,activation="relu",kernel_regularizer="l2")(x)
    end_layer=tf.keras.layers.Dense(classes,activation="softmax")(x)
    model=tf.keras.Model(inputs=inputs,outputs=end_layer)
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss="categorical_crossentropy",metrics=["accuracy"])

    return model


#model for the energy prediction
#this function is almost identical to multi_tel, but it changes the last layer with


def model_multi_tel_energy(len_inputs=4,input_shapes=[(55,93,1)],learning_rate=1e-5,pre_proces_model=None,batch_init=False,filtros=None,last_dense=None,autoencoder=None,common_pre=True):
    #model for the energy prediction
    inputs=[]
    outputs=[[] for i in range(len_inputs)]
    if len(input_shapes)==1:
        for i in range(1,len_inputs):
            input_shapes.append(input_shapes[0])

    for i in range(len_inputs):
        inputs.append(tf.keras.Input(shape=input_shapes[i]))

    if pre_proces_model:
        for i in range(len_inputs):
            outputs[i]=pre_proces_model(inputs[i],first_model=autoencoder)
    else:
        if filtros is None:
            filtros=[[64,128],[128,254,64],[32]]
        #si no le metemos un modelo pues habra que meter aqui chicha porque sino se queda esto muy vacio
        if common_pre:
            pre_model=model_1_tel(input_shapes[0],batch_init=batch_init,filtros=filtros,first_part=True,first_model=autoencoder)
            outputs[0]=pre_model(inputs[0])
            for i in range(1,len_inputs):
                if input_shapes[i]!=input_shapes[i-1]:
                #esto esta suponiendo que ponemos juntos los que tienen igual shape
                    pre_model=model_1_tel(input_shapes[i],batch_init=batch_init,filtros=filtros,first_part=True,first_model=autoencoder) 
                outputs[i]=pre_model(inputs[i])

        else:
            for i in range(len_inputs):
                pre_model=model_1_tel(input_shape=input_shapes[i],batch_init=batch_init,filtros=filtros,first_part=True)
                outputs[i]=pre_model(inputs[i]) 
    #nos falta la ultima parte
    if last_dense is None:
        last_dense=[65,25]
    x=tf.keras.layers.concatenate(outputs)
    for i in last_dense:
        x=tf.keras.layers.Dense(i,activation="relu")(x)
    end_layer=tf.keras.layers.Dense(1)(x)
    model=tf.keras.Model(inputs=inputs,outputs=end_layer)
    #model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),loss="mse")

    return model

#How to use it?

"""
model=model_multi_tel(learning_rate=1e-4,last_dense=[150,50],filtros=[[32, 64], [128, 256],
                        [64, 32,4]],input_shapes=[(55,93,1),(55,93,1),(55,93,1),(55,93,1)],
                        batch_init=False,common_pre=True)
"""