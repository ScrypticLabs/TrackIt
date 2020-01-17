from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import random
import os
import matplotlib.pyplot as plt
import numpy as np

class Autoencoder_Status(Callback):
    def __init__(self, X, Y):
        self.X = X
        self.Y = Y

    def on_train_begin(self, logs={}):
        return
 
    def on_train_end(self, logs={}):
        return
 
    def on_epoch_begin(self, epoch, logs={}):
        return
 
    def on_epoch_end(self, epoch, logs={}):
        if epoch % 5 == 0:
            sample = random.randint(0, self.X.shape[0]-1)
            x = self.X[sample:sample+2]
            y = self.Y[sample:sample+2]
            y_hat = self.model.predict(x)
            plt.subplot(1,2,1)
            plt.axis('off')
            plt.margins(0,0)
            plt.tight_layout(pad=1)
            plt.imshow(x[0])
            plt.subplot(1,2,2)
            plt.axis('off')
            plt.margins(0,0)
            plt.tight_layout(pad=1)
            plt.imshow(y_hat[0])
            plt.show()

    def on_batch_begin(self, batch, logs={}):
        return
 
    def on_batch_end(self, batch, logs={}):
        return

class Autoencoder():
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.autoencoder = None
        self.encoder_layers = []
        self.decoder_layers = []
        self.status_callback = None

        # self.model_weights_path = os.getcwd()+"/../models/autoencoder.hdf5"
        self.checkpoint = None
        
    def build_model(self, input_dim, latent_dim):
        e =  [Input(shape=input_dim)]
        e += [Conv2D(64, (4, 4), strides=(2,2), activation='relu', padding='same')(e[-1])]
        e += [Conv2D(64, (4,4), strides=(2,2), activation='relu', padding='same')(e[-1])]
        e += [Conv2D(64, (4,4), strides=(1,1), activation='relu', padding='same')(e[-1])]
        e += [Flatten()(e[-1])]
        e += [Dense(units=64)(e[-1])]
        
        # at this point the representation should match the latent shape

        d = [Dense(units=8*8*3, activation='relu')(e[-1])]
        d += [Reshape((8, 8, 3))(d[-1])]
        d += [Conv2DTranspose(64, (4,4), strides=(2,2), activation='relu', padding='same')(d[-1])]
        d += [Conv2DTranspose(64, (4,4), strides=(2,2), activation='relu', padding='same')(d[-1])]
        d += [Conv2DTranspose(64, (4,4), strides=(1,1), activation='relu', padding='same')(d[-1])]
        d += [Flatten()(d[-1])]
        d += [Dense(units=32*32*3, activation='sigmoid')(d[-1])]
        d += [Reshape((32, 32, 3))(d[-1])]
        
        self.autoencoder = Model(e[0], d[-1])
        self.save_encoder_layers(autoencoder=self.autoencoder, input_dim=input_dim, encoder_layers=e, decoder_layers=d)
        self.save_decoder_layers(autoencoder=self.autoencoder, input_dim=latent_dim, encoder_layers=e, decoder_layers=d)
        self.encoder = Model(self.encoder_layers[0], self.encoder_layers[-1])
        self.decoder = Model(self.decoder_layers[0], self.decoder_layers[-1])

        def custom_loss(y_true, y_pred):
            y_true_flat = K.reshape(y_true, (-1, 32*32*3))
            y_pred_flat = K.reshape(y_pred, (-1, 32*32*3))
            img_loss = K.sum(K.square(y_true_flat-y_pred_flat), axis=1)
            loss = K.mean(img_loss)
            return loss

        self.autoencoder.compile(optimizer=Adam(lr=0.0001), loss=custom_loss)

    # extracts the layers from the autoencoder Model to build the encoder
    def save_encoder_layers(self, autoencoder, input_dim, encoder_layers, decoder_layers):
        self.encoder_layers = [Input(shape=input_dim)]
        INPUT = self.encoder_layers[0]
        for i in range(1, len(encoder_layers)):
            OUTPUT = autoencoder.layers[i](INPUT)
            self.encoder_layers += [OUTPUT]
            INPUT = OUTPUT
        return self.encoder_layers
    
    # extracts the layers from the autoencoder Model to build the decoder
    def save_decoder_layers(self, autoencoder, input_dim, encoder_layers, decoder_layers):
        self.decoder_layers = [Input(shape=input_dim)]
        INPUT = self.decoder_layers[0]
        for i in range(len(encoder_layers), len(encoder_layers)+len(decoder_layers)):
            OUTPUT = autoencoder.layers[i](INPUT)
            self.decoder_layers += [OUTPUT]
            INPUT = OUTPUT
        return self.decoder_layers

    def train(self, X_train, Y_train, X_val, Y_val, epochs=1, batch_size=64, shuffle=True):
        # if self.status_callback == None:
            # self.status_callback = Autoencoder_Status(X_train, Y_train)
        # self.checkpoint = ModelCheckpoint(self.model_weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        history = self.autoencoder.fit(
                    X_train, 
                    Y_train,
                    epochs=epochs, 
                    batch_size=batch_size,
                    shuffle=shuffle, 
                    validation_data=(X_val, Y_val)
                    # callbacks=[self.status_callback, self.checkpoint]
                )
        
        return history
    
    def save_weights(self, filepath):
        self.autoencoder.save_weights(filepath)
    
    def set_weights(self, filepath):
        self.autoencoder.load_weights(filepath)

    def encode_series(self, X):
        embeddings = []
        for i in range(X.shape[0]):
            embedding = self.encoder.predict(X[i])
            embeddings.append(embedding)
        return np.array(embeddings)
    
    def decode_series(self, X):
        reconstructions = []
        for i in range(X.shape[0]):
            reconstruction = self.decoder.predict(X[i])
            reconstructions.append(reconstruction)
        return np.array(reconstructions)       