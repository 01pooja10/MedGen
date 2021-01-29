from keras.layers import GRU
from keras.layers import Embedding
from tensorflow import keras
from keras.optimizers import Adam
from keras.losses import SparseCategoricalCrossentropy
import keras
from keras_self_attention import SeqSelfAttention
from keras.backend import clear_session

clear_session()

#encoder
input_1=Input(shape=(1,2048),name="Images")
encoder_out=Dense(512,activation="relu",name="enc_dense")(input_1)


#decoder
input_text=Input(shape=(max_len),name="text")

embedding_out=tf.keras.layers.Embedding(input_dim=vocab_size,output_dim=300,input_length=max_len,mask_zero=True,trainable=False,weights=[embedding_matrix])(input_text)

lstm_out= LSTM(units=128, activation='tanh', recurrent_activation='sigmoid', use_bias=True,
            kernel_initializer=tf.keras.initializers.glorot_uniform(seed=23),
            recurrent_initializer=tf.keras.initializers.orthogonal(seed=7),
            bias_initializer=tf.keras.initializers.zeros(), return_sequences=True, name="LSTM1")(embedding_out)

#attention layer
attn=SeqSelfAttention(units=256,attention_width=15,name='Attention')(lstm_out)

x=Dense(512,kernel_initializer=tf.keras.initializers.he_normal(seed=1),activation="relu")(attn)

x1=Dropout(0.25)(x)

x1=Dense(vocab_size,activation="softmax")(x1)

#attention model
attention=Model(inputs=[input_1,input_text],outputs=x1)
attention.summary()

#keras.utils.plot_model(attention)

loss_function = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False, reduction='auto')
def maskedLoss(y_true, y_pred):
    #print('y_predicted: ',y_pred.shape)
    #print('y_true: ',y_true.shape)
    mask = tf.math.logical_not(tf.math.equal(y_true, 0))
    loss_ = loss_function(y_true, y_pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ = loss_*mask
    loss_ = tf.reduce_mean(loss_)
    return loss_

attention.compile(optimizer=Adam(learning_rate=0.001),loss=maskedLoss)

import warnings
warnings.filterwarnings('ignore')

train_loss_list = []

for epoch in range(15):
    print('EPOCH : ',epoch+1)
    batch_loss_train = 0

    for img, report in train_dataset:
        r1 = [word.decode('utf-8') for word in np.array(report)]
        img_input, rep_input, output_word = load_data(img.numpy(), r1)
        rep_input = tf.keras.preprocessing.sequence.pad_sequences(rep_input, maxlen=80, padding='post')

        loss = attention.train_on_batch([img_input,rep_input],[rep_input,output_word])

        batch_loss_train += loss

    train_loss = batch_loss_train/(len(y_train)//15)
    print('Training Loss: {}'.format(train_loss))
    
attention.save_weights('attention.h5')
