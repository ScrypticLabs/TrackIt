from keras.models import Sequential
from keras.layers import Input, Dense, LSTM, TimeDistributed, RepeatVector
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras import backend as K
from keras.callbacks import TensorBoard, ModelCheckpoint, Callback

class Seq2Seq():
    def __init__(self):
        self.encoder = None
        self.decoder = None
        self.model = None
        self.encoder_layers = []
        self.decoder_layers = []
        self.status_callback = None

        # self.model_weights_path = os.getcwd()+"/../models/seq2seq.hdf5"
        self.checkpoint = None
        
    def build_model(self, input_length, input_dim, latent_dim, output_length, output_dim):
        e =  [Input(shape=(input_length, input_dim))]
        e += [LSTM(1024, input_shape=(input_length, input_dim), return_sequences=True)(e[-1])]
        e += [LSTM(512, input_shape=(input_length, 1024), return_sequences=True)(e[-1])]
        e += [LSTM(256, input_shape=(input_length, 512))(e[-1])]

        d =  [RepeatVector(output_length)(e[-1])]
        d += [LSTM(512, input_shape=(output_length, 256), return_sequences=True)(d[-1])]
        d += [LSTM(1024, input_shape=(output_length, 512), return_sequences=True)(d[-1])]
        d += [TimeDistributed(Dense(output_dim, activation='linear'))(d[-1])]
        # d += [LSTM(output_dim, input_shape=(output_length, 1024), return_sequences=True)(d[-1])]
        
        self.model = Model(e[0], d[-1])
        self.save_encoder_layers(model=self.model, input_length=input_length, input_dim=input_dim, encoder_layers=e, decoder_layers=d)
        self.save_decoder_layers(model=self.model, latent_dim=latent_dim, encoder_layers=e, decoder_layers=d)
        
        self.encoder = Model(self.encoder_layers[0], self.encoder_layers[-1])
        self.decoder = Model(self.decoder_layers[0], self.decoder_layers[-1])

        def custom_loss(y_true, y_pred):
            diff = K.square(y_true-y_pred)
            result = K.sum(diff, axis=2)
            loss = K.mean(result)
            # diff_flat = K.reshape(diff, (-1, output_length*output_dim))
            # loss = K.sum(diff_flat)
            return loss

        self.model.compile(optimizer=Adam(lr=0.0001), loss=custom_loss)

    # extracts the layers from the seq2seq Model to build the encoder
    def save_encoder_layers(self, model, input_length, input_dim, encoder_layers, decoder_layers):
        self.encoder_layers = [Input(shape=(input_length, input_dim))]
        INPUT = self.encoder_layers[0]
        for i in range(1, len(encoder_layers)):
            OUTPUT = model.layers[i](INPUT)
            self.encoder_layers += [OUTPUT]
            INPUT = OUTPUT
        return self.encoder_layers
    
    # extracts the layers from the seq2seq Model to build the decoder
    def save_decoder_layers(self, model, latent_dim, encoder_layers, decoder_layers):
        self.decoder_layers = [Input(shape=latent_dim)]
        INPUT = self.decoder_layers[0]
        for i in range(len(encoder_layers), len(encoder_layers)+len(decoder_layers)):
            OUTPUT = model.layers[i](INPUT)
            self.decoder_layers += [OUTPUT]
            INPUT = OUTPUT
        return self.decoder_layers

    def train(self, X_train, Y_train, X_val, Y_val, epochs=1, batch_size=64):
        # if self.status_callback == None:
            # self.status_callback = Autoencoder_Status(X_train, Y_train)
        # self.checkpoint = ModelCheckpoint(self.model_weights_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
        history = self.model.fit(
                    X_train, 
                    Y_train,
                    epochs=epochs, 
                    batch_size=batch_size,
                    validation_data=(X_val, Y_val)
                    # callbacks=[self.status_callback, self.checkpoint]
                )
        return history
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
    
    def set_weights(self, filepath):
        self.model.load_weights(filepath)