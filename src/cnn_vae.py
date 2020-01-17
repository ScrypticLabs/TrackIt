from keras.layers import Input, Lambda, Dense, Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Flatten, Reshape
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback
import random
import os
import matplotlib.pyplot as plt

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
        self.mu, self.log_var = None, None

        self.encoder = None
        self.encoder_mu_log_var = None
        self.decoder = None
        self.autoencoder = None
        self.status_callback = None

        # self.model_weights_path = os.getcwd()+"/../models/autoencoder.hdf5"
        self.checkpoint = None
        
    def build_model(self, input_dim, latent_dim):
        # ENCODER
        encoder_input = Input(shape=input_dim)
        x = Conv2D(64, (4, 4), strides=(2,2), activation='relu', padding='same')(encoder_input)
        x = Conv2D(64, (4,4), strides=(2,2), activation='relu', padding='same')(x)
        x = Conv2D(64, (4,4), strides=(1,1), activation='relu', padding='same')(x)
        x = Flatten()(x)
        self.mu, self.log_var = Dense(units=64)(x), Dense(units=64)(x)
        self.encoder_mu_log_var = Model(encoder_input, (self.mu, self.log_var))
        def sampling(args):
            mu, log_var = args
            epsilon = K.random_normal(shape=K.shape(mu), mean=0., stddev=1.0)
            return mu + K.exp(log_var / 2) * epsilon
        encoder_output = Lambda(sampling)([self.mu, self.log_var])
        self.encoder = Model(encoder_input, encoder_output)

        # DECODER
        decoder_input = Input(shape=latent_dim)
        x = Dense(units=8*8*3, activation='relu')(decoder_input)
        x = Reshape((8, 8, 3))(x)
        x = Conv2DTranspose(64, (4,4), strides=(2,2), activation='relu', padding='same')(x)
        x = Conv2DTranspose(64, (4,4), strides=(2,2), activation='relu', padding='same')(x)
        x = Conv2DTranspose(64, (4,4), strides=(1,1), activation='relu', padding='same')(x)
        x = Flatten()(x)
        x = Dense(units=32*32*3, activation='sigmoid')(x)
        x = Reshape((32, 32, 3))(x)
        decoder_output = x
        self.decoder = Model(decoder_input, decoder_output)

        # AUTOENCODER
        autoencoder_input = encoder_input
        autoencoder_output = self.decoder(encoder_output)
        self.autoencoder = Model(autoencoder_input, autoencoder_output)

        def custom_loss(y_true, y_pred):
            y_true_flat = K.reshape(y_true, (-1, 32*32*3))
            y_pred_flat = K.reshape(y_pred, (-1, 32*32*3))
            img_loss = K.sum(K.square(y_true_flat-y_pred_flat), axis=1)
            loss = K.mean(img_loss)
            return loss
        
        def kl_loss(y_true, y_pred):
            kl_loss =  -0.5 * K.sum(1 + self.log_var - K.square(self.mu) - K.exp(self.log_var), axis = 1)
            return kl_loss
        
        def loss(y_true, y_pred):
            return custom_loss(y_true, y_pred) + kl_loss(y_true, y_pred)

        self.autoencoder.compile(optimizer=Adam(lr=0.0001), loss=loss)

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