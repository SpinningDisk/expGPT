import keras
import tensorflow as tf
from math import ceil


@tf.keras.utils.register_keras_serializable()
class Decoder(keras.layers.Layer):
    def __init__(self, seq_length: int, embedding_size: int, heads: int, hidden_size: int, key_dim: int = None, dropout_rate: float = 0.1, attention_dropout: float = 0.1, activation: str = "relu", causal: bool = True, use_layernorm: bool = True, use_flashattention: bool = True, residual: bool = True, kernel_initializer = "glorot_uniform", trainable: bool=True, **kwargs):
        super().__init__(**kwargs)
        self.seq_length = seq_length
        self.embedding_size = embedding_size
        self.heads = heads
        self.hidden_size = hidden_size
        self.key_dim = key_dim
        self.dropout_rate = dropout_rate
        self.attention_dropout = attention_dropout
        self.activation = activation
        self.causal = causal
        self.use_layernorm = use_layernorm
        self.use_flashattention = use_flashattention
        self.residual = residual
        self.kernel_initializer = kernel_initializer
        self.trainable = trainable
    def build(self, input_shape):
        match(self.causal):
            case True:
                self.mask = tf.linalg.band_part(tf.ones((self.seq_length, self.seq_length)), -1, 0)
            case default:
                self.mask = tf.ones((self.seq_length, self.seq_length))
        match(self.key_dim):
            case None:
                self.key_dim = ceil(self.embedding_size/self.heads)
            case default:
                self.key_dim = self.key_dim
        
        self.ffn = keras.Sequential([keras.layers.Dense(units=4*self.embedding_size, activation=self.activation, kernel_initializer=self.kernel_initializer, name=f"ffn1_d{x}") for x in range(self.hidden_size)] + [keras.layers.Dense(units=self.embedding_size, activation=self.activation, kernel_initializer=self.kernel_initializer, name="ffn1_df")])
        self.ffn.build(input_shape)
        self.add = keras.layers.Add(name="addition1")
#        self.add.build(input_shape)
        self.dropoutLayer = keras.layers.Dropout(self.dropout_rate, name="dropoutLayer")
        self.dropoutLayer.build(input_shape)

        self.inputNorm = keras.layers.LayerNormalization(name="inputNorm")
        self.inputNorm.build(input_shape)
        self.mmha=keras.layers.MultiHeadAttention(
            num_heads=self.heads,
            key_dim=self.key_dim,
            value_dim=None,
            dropout=self.attention_dropout,
            use_bias=True,
            output_shape=None,
            attention_axes=None,
            flash_attention=self.use_flashattention,
            kernel_initializer=self.kernel_initializer,
            bias_initializer="zeros",
            name="mmha",
        )
        self.attentionNorm = keras.layers.LayerNormalization(name="attentionNorm")
        self.attentionNorm.build(input_shape)
        super().build(input_shape)
    def call(self, inputs):
        inputNorm = self.inputNorm(inputs) if self.use_layernorm else inputs
        mmha = self.mmha(query=inputNorm, key=inputNorm, value=inputNorm, attention_mask=self.mask)
        residualAttention = self.add([inputs, mmha]) if self.residual else mmha
        residualAttention = self.dropoutLayer(residualAttention)
        attentionNorm = self.attentionNorm(residualAttention) if self.use_layernorm else residualAttention
        ffnOut = self.ffn(attentionNorm)
        ffnOut = self.dropoutLayer(ffnOut)
        residualFFN = self.add([attentionNorm, ffnOut]) if self.residual else ffnOut
        return residualFFN
    def compute_output_shape(self, inputs):
        return inputs
    def get_config(self):
        config = super().get_config()
        config.update({
            "mask": keras.saving.serialize_keras_object(self.mask),
            "ffn": keras.saving.serialize_keras_object(self.ffn),
            "add": keras.saving.serialize_keras_object(self.add),
            "dropoutLayer": keras.saving.serialize_keras_object(self.dropoutLayer),
            "inputNorm": keras.saving.serialize_keras_object(self.inputNorm),
            "mmha": keras.saving.serialize_keras_object(self.mmha),
            "attentionNorm": keras.saving.serialize_keras_object(self.attentionNorm),
            "seq_length": self.seq_length,
            "embedding_size": self.embedding_size,
            "heads": self.heads,
            "hidden_size": self.hidden_size,
            "key_dim": self.key_dim,
            "dropout_rate": self.dropout_rate,
            "attention_dropout": self.attention_dropout,
            "activation": self.activation,
            "causal": self.causal,
            "use_layernorm": self.use_layernorm,
            "use_flashattention": self.use_flashattention,
            "residual": self.residual,
            "kernel_initializer": self.kernel_initializer,
            "trainable": self.trainable
            
        })
        return config
    @classmethod
    def from_config(cls, config):
        mask = keras.saving.deserialize_keras_object(config.pop("mask"))
        ffn = keras.saving.deserialize_keras_object(config.pop("ffn"))
        add = keras.saving.deserialize_keras_object(config.pop("add"))
        dropoutLayer = keras.saving.deserialize_keras_object(config.pop("dropoutLayer"))
        inputNorm = keras.saving.deserialize_keras_object(config.pop("inputNorm"))
        mmha = keras.saving.deserialize_keras_object(config.pop("mmha"))
        attentionNorm = keras.saving.deserialize_keras_object(config.pop("attentionNorm"))

        obj = cls(**config)
        obj.mask = mask
        obj.ffn = ffn
        obj.add = add
        obj.dropoutLayer = dropoutLayer
        obj.inputNorm = inputNorm
        obj.mmha = mmha
        obj.attentionNorm = attentionNorm
        return cls(**config)
@tf.keras.utils.register_keras_serializable()
class Encoder(keras.layers.Layer):
    def __init__(self, seq_length: int, embedding_size: int, heads: int, hidden_size: int, key_dim: int = None, dropout_rate: float = 0.1, attention_dropout: float = 0.1, activation: str = "relu", use_layernorm: bool = True, use_flashattention: bool = True, residual: bool = True, kernel_initializer = "glorot_uniform"):
        super().__init__()
       
        match(key_dim):
            case None:
                key_dim = ceil(embedding_size/heads)
            case default:
                key_dim = key_dim
        self.use_layernorm = use_layernorm
        self.residual = residual
        
        self.ffn = keras.Sequential([keras.layers.Dense(units=4*embedding_size, activation=activation, kernel_initializer=kernel_initializer) for _ in range(hidden_size)] + [keras.layers.Dense(units=embedding_size, activation=activation, kernel_initializer=kernel_initializer)])
        self.add = keras.layers.Add()
        self.dropoutLayer = keras.layers.Dropout(dropout_rate)

        self.inputNorm = keras.layers.LayerNormalization()
        self.mmha=keras.layers.MultiHeadAttention(
            num_heads=heads,
            key_dim=key_dim,
            value_dim=None,
            dropout=attention_dropout,
            use_bias=True,
            output_shape=None,
            attention_axes=None,
            flash_attention=use_flashattention,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
        )
        self.attentionNorm = keras.layers.LayerNormalization()

    def build(self):
        pass
    def call(self, inputs):
        inputNorm = self.inputNorm(inputs) if self.use_layernorm else inputs
        mmha = self.mmha(query=inputNorm, key=inputNorm, value=inputNorm)
        residualAttention = self.add([inputs, mmha]) if self.residual else mmha
        residualAttention = self.dropoutLayer(residualAttention)
        attentionNorm = self.attentionNorm(residualAttention) if self.use_layernorm else residualAttention
        ffnOut = self.ffn(attentionNorm)
        ffnOut = self.dropoutLayer(ffnOut)
        residualFFN = self.add([attentionNorm, ffnOut]) if self.residual else ffnOut
        return residualFFN
    def compute_output_shape(self, inputs):
        return inputs
@tf.keras.utils.register_keras_serializable()
class EnDecoder(keras.layers.Layer):
    def __init__(self, seq_length_decoder: int, seq_length_encoder: int, embedding_size: int, heads_encoder: int, heads_decoder1: int, heads_decoder2: int, hidden_size: int, key_dim: int = None, dropout_rate: float = 0.1, attention_dropout: float = 0.1, activation: str = "relu", causal: bool = True, use_layernorm: bool = True, use_flashattention: bool = True, residual: bool = True, kernel_initializer = "glorot_uniform"):
        super().__init__()
        
        # Decoder:
        self.decoder_inputs = keras.layers.Lambda(lambda x: x[:, 1])
        match(causal):
            case True:
                self.mask = tf.linalg.band_part(tf.ones((seq_length_decoder, seq_length_decoder)), -1, 0)
            case default:
                self.mask = tf.ones((seq_length_decoder, seq_length_decoder))
        match(key_dim):
            case None:
                key_dim = ceil(embedding_size/heads_decoder1)
            case default:
                key_dim = key_dim
        self.use_layernorm = use_layernorm
        self.residual = residual
        
        self.ffn = keras.Sequential([keras.layers.Dense(units=4*embedding_size, activation=activation, kernel_initializer=kernel_initializer) for _ in range(hidden_size)] + [keras.layers.Dense(units=embedding_size, activation=activation, kernel_initializer=kernel_initializer)])
        self.add = keras.layers.Add()
        self.dropoutLayer = keras.layers.Dropout(dropout_rate)

        self.inputNorm = keras.layers.LayerNormalization()
        self.mmha=keras.layers.MultiHeadAttention(
            num_heads=heads_decoder1,
            key_dim=key_dim,
            value_dim=None,
            dropout=attention_dropout,
            use_bias=True,
            output_shape=None,
            attention_axes=None,
            flash_attention=use_flashattention,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
        )
        self.attentionNorm = keras.layers.LayerNormalization()
        self.mha=keras.layers.MultiHeadAttention(
            num_heads=heads_decoder2,
            key_dim=key_dim,
            value_dim=None,
            dropout=attention_dropout,
            use_bias=True,
            output_shape=None,
            attention_axes=None,
            flash_attention=use_flashattention,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
        )
        self.encoderAttendedNorm = keras.layers.LayerNormalization()

        # Encoder:
        self.encoder_inputs = keras.layers.Lambda(lambda x: x[:, 0])
        match(key_dim):
            case None:
                enc_key_dim = ceil(embedding_size/heads)
            case default:
                enc_key_dim = key_dim
        self.enc_use_layernorm = use_layernorm
        self.enc_residual = residual
        
        self.enc_ffn = keras.Sequential([keras.layers.Dense(units=4*embedding_size, activation=activation, kernel_initializer=kernel_initializer) for _ in range(hidden_size)] + [keras.layers.Dense(units=embedding_size, activation=activation, kernel_initializer=kernel_initializer)])
        self.enc_add = keras.layers.Add()
        self.enc_dropoutLayer = keras.layers.Dropout(dropout_rate)

        self.enc_Input = keras.layers.Input((seq_length_encoder, embedding_size))
        self.enc_inputNorm = keras.layers.LayerNormalization()
        self.enc_mmha=keras.layers.MultiHeadAttention(
            num_heads=heads_encoder,
            key_dim=key_dim,
            value_dim=None,
            dropout=attention_dropout,
            use_bias=True,
            output_shape=None,
            attention_axes=None,
            flash_attention=use_flashattention,
            kernel_initializer=kernel_initializer,
            bias_initializer="zeros",
        )
        self.enc_attentionNorm = keras.layers.LayerNormalization()
    def build(self, inputs):
        assert len(inputs)==2
        pass
    def call(self, inputs):
        #Encoder:
        encoder_inputs = inputs[0]
        encoder_inputNorm = self.enc_inputNorm(encoder_inputs)if self.use_layernorm else encoder_inputs
        encoder_mmha = self.enc_mmha(query=encoder_inputNorm, key=encoder_inputNorm, value=encoder_inputNorm)
        encoder_residualAttention = self.add([encoder_inputs, encoder_mmha]) if self.residual else encoder_mmha
        encoder_residualAttention = self.dropoutLayer(encoder_residualAttention)
        encoder_attentionNorm = self.enc_attentionNorm(encoder_residualAttention) if self.use_layernorm else encoder_residualAttention
        encoder_ffnOut = self.enc_ffn(encoder_attentionNorm)
        encoder_ffnOut = self.dropoutLayer(encoder_ffnOut)
        encoderOut = self.add([encoder_attentionNorm, encoder_ffnOut]) if self.residual else encoder_ffnOut
        
        decoder_inputs = inputs[1]
        decoder_inputNorm = self.inputNorm(decoder_inputs) if self.use_layernorm else decoder_inputs
        decoder_mmha = self.mmha(query=decoder_inputNorm, key=decoder_inputNorm, value=decoder_inputNorm, attention_mask=self.mask)
        decoder_residualAttention = self.add([decoder_inputNorm, decoder_mmha]) if self.residual else decoder_mmha
        decoder_residualAttention = self.dropoutLayer(decoder_residualAttention)
        decoder_attentionNorm = self.attentionNorm(decoder_residualAttention) if self.use_layernorm else decoder_residualAttention
        decoder_encoder_attention = self.mha(query=decoder_attentionNorm, key=encoderOut, value=encoderOut, attention_mask=None)
        decoder_encoder_residualAttention = self.add([decoder_encoder_attention, decoder_attentionNorm])
        decoder_encoder_residualAttention = self.dropoutLayer(decoder_encoder_residualAttention)
        decoder_encoder_attentionNorm = self.encoderAttendedNorm(decoder_encoder_residualAttention)
        decoder_ffnOut = self.ffn(decoder_encoder_attentionNorm)
        decoder_ffnOut = self.dropoutLayer(decoder_ffnOut)
        decoder_residualFFN = self.add([decoder_encoder_attentionNorm, decoder_ffnOut]) if self.residual else decoder_ffnOut
        return decoder_residualFFN
    def compute_output_shape(self, inputs):
        return inputs[1]


if __name__=="__main__":
    import numpy as np
    decoder = Decoder(embedding_size=128, heads=16, hidden_size=4, seq_length=512, use_flashattention=False)
    def testDecoder(embedding_size, seq_length):
        inputs = keras.layers.Input((seq_length, embedding_size))
        trans1 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(inputs)
        trans2 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans1)
        trans3 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans2)
        trans4 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans3)
        trans5 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans4)
        trans6 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans5)
        trans7 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans6)
        trans8 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans7)
        trans9 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans8)
        trans10 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans9)
        trans11 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans10)
        trans12 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans11)
        trans13 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans12)
        trans14 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans13)
        trans15 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans14)
        trans16 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans15)
        trans17 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans16)
        trans18 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans17)
        trans19 = Decoder(embedding_size=embedding_size, heads=16, hidden_size=3, seq_length=seq_length, use_flashattention=False, activation="swish")(trans18)
        out= keras.layers.Dense(units=128, activation="swish")(trans19)
        return keras.Model(inputs=inputs, outputs=out)
    def testEncoder():
        inputs = keras.layers.Input((512, 128*2))
        trans1 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(inputs)
        trans2 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans1)
        trans3 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans2)
        trans4 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans3)
        trans5 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans4)
        trans6 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans5)
        trans7 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans6)
        trans8 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans7)
        trans9 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans8)
        trans10 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans9)
        trans11 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans10)
        trans12 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans11)
        trans13 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans12)
        trans14 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans13)
        trans15 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans14)
        trans16 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans15)
        trans17 = Encoder(embedding_size=128*2, heads=16, hidden_size=3, seq_length=512, use_flashattention=False, activation="swish")(trans16)
        out = keras.layers.Dense(units=128*2, activation="swish")(trans17)
        return keras.Model(inputs=inputs, outputs=out)
    def testEnDecoder():
        expert = keras.layers.Input((512*4, 128))
        goal = keras.layers.Input((512, 128))
        trans1 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, goal])
        trans2 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans1])
        trans3 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans2])
        trans4 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans3])
        trans5 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans4])
        trans6 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans5])
        trans7 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans6])
        trans8 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans7])
        trans9 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans8])
        trans10 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans9])
        trans11 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans10])
        trans12 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans11])
        trans13 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans12])
        trans14 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans13])
        trans15 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans14])
        trans16 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans15])
        trans17 = EnDecoder(embedding_size=128, heads_encoder=16, heads_decoder1=32, heads_decoder2=32, hidden_size=4, seq_length_decoder=512, seq_length_encoder=512*4, use_flashattention=False, activation="swish")([expert, trans16])
        out = keras.layers.Dense(units=128, activation="swish")(trans17)
        return keras.Model(inputs=[expert, goal], outputs=out)
    import time
    start = time.time()
    samples=100
    model = testDecoder(8, 16)
    model.save("test.keras")
    model = keras.models.load_model("test.keras", custom_objects={"Decoder": Decoder})

    for i in range(samples):
        out = model([np.expand_dims(np.random.rand(512*4, 128), 0), np.expand_dims(np.random.rand(16, 8), 0)][1])
        model.save("test.keras")
        model = keras.models.load_model("test.keras", custom_objects={"Decoder": Decoder})
        print(out)
        print(out.shape)
    end = time.time()
    print(f"finished {samples} transformations in {end-start}s ({((end-start)/samples)/1000}ms/sample on average)")
