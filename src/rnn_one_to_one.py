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
        self.forward = None
        self.encoder_layers = []
        self.decoder_layers = []
        self.status_callback = None

        # self.model_weights_path = os.getcwd()+"/../models/seq2seq.hdf5"
        self.checkpoint = None
        
    def build_model(self, input_dim, output_dim):
        rnn_in = Input(shape=(None, input_dim))
        lstm = LSTM(1024, input_shape=(None, input_dim), return_sequences=True, return_state=True)
        rnn_out = Dense(output_dim, activation='linear')

        x, _, _ = lstm(rnn_in) 
        x = rnn_out(x)
        self.model = Model(rnn_in, x)

        state_input_h, state_input_c = Input(shape=(1024,)), Input(shape=(1024,))
        x, state_h, state_c = lstm(rnn_in, initial_state=[state_input_h, state_input_c])
        x = rnn_out(x)

        self.forward = Model([rnn_in]+[state_input_h, state_input_c], [x, state_h, state_c])

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
                    validation_data=(X_val, Y_val),
                    shuffle=False
                    # callbacks=[self.status_callback, self.checkpoint]
                )
        return history
    
    def save_weights(self, filepath):
        self.model.save_weights(filepath)
    
    def set_weights(self, filepath):
        self.model.load_weights(filepath)