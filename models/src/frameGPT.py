from keras import layers, Model
import sys
sys.path.insert(0, "/mnt/data/dev/ML/lib/")
from SelectiveTransformation import SelectiveTransformation
from Transformers import EnDecoder

class GPT():
    def __init__(self):
        self.screen_size = (18, 18)
        self.seq_length_decoder = self.screen_size[0]*self.screen_size[1]
        self.embedding_size = 128
        self.seq_length_encoder = 32
        self.game_collection = 100
        self.heads_encoder = 8
        self.heads_decoder1 = 8
        self.heads_decoder2 = 8
        self.hidden_size = 4
        self.activation = "swish"
        self.use_flashattention = False
    def GPT(self):
#        inputs_encoder_embedding = layers.Input((self.seq_length_encoder, self.embedding_size))
        inputs_encoder = layers.Input((self.seq_length_encoder, self.embedding_size))
        inputs_encoder_embedding = SelectiveTransformation(keys=[[x for _ in range(self.embedding_size)] for x in range(self.seq_length_encoder)], input_size=(self.seq_length_encoder, self.embedding_size), key_shape=(self.seq_length_encoder, self.embedding_size))(inputs_encoder)

        inputs_decoder = layers.Input((self.seq_length_decoder, self.embedding_size))
        

        block1 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, inputs_decoder])
        block2 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block1])
        block3 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block2])
        block4 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block3])
        block5 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block4])
        block6 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block5])
        block7 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block6])
        block8 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block7])
        block9 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block8])
        block10 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block9])
        block11 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block10])
        block12 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block11])
        block13 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block12])
        block14 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block13])
        block15 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block14])
        block16 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block15])
        block17 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block16])
        block18 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block17])
        block19 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block18])
        block20 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block19])
        block21 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block20])
        block22 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block21])
        block23 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block22])
        block24 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block23])
        block25 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block24])
        block26 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block25])
        block27 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block26])
        block28 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block27])
        block29 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block28])
        block30 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block29])
        block31 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block30])
        block32 = EnDecoder(seq_length_decoder=self.seq_length_decoder, seq_length_encoder=self.seq_length_encoder, embedding_size=self.embedding_size, heads_encoder=self.heads_encoder, heads_decoder1=self.heads_decoder1, heads_decoder2=self.heads_decoder2, hidden_size=self.hidden_size, activation=self.activation, use_flashattention=self.use_flashattention)([inputs_encoder_embedding, block31])

        out = layers.Dense(units=self.embedding_size, activation=self.activation)(block32)
        return Model(inputs=[inputs_encoder, inputs_decoder], outputs=out)

if __name__=="__main__":
    sys.path.insert(0, "/mnt/data/dev/ML/environments/snek")
    from  env import SnakeEnv
    env = SnakeEnv(token_size=128)
    env.reset()

    import numpy as np
    from PIL import Image
    import random
    baseGPT = GPT()
    myGPT=baseGPT.GPT()
    samples = 100
    import time
    start = time.time()
    for i in range(samples):
        env_response = np.array(env._make_frame()).reshape(324, 128)
        env.render()
        if env.step([random.randint(-1, 1), random.randint(-1, 1)])[2] == True:
            env.reset()
        in_array = [np.expand_dims(np.array([[env.game_id for _ in range(baseGPT.embedding_size)] for x in range(baseGPT.seq_length_encoder)]), 0), np.expand_dims(env_response, 0)]
        out = myGPT(in_array)
        out = out.numpy()
        out_image = Image.fromarray((out.reshape((out.shape[1], out.shape[2]))*255).astype(np.uint8))
        out_image.save(f"frames/frame{i}outputs.png")
        board = env_response.reshape(18, 18, 128)*50
        board_image = Image.fromarray(board[:, :, 1].astype(np.uint8))
        print(board[:, :, 1])
        board_image.save(f"frames/frame{i}boardss.png")
        print(f"{out} at index {i}")
    end = time.time()
    print(f"running {samples} sample(s) took {end-start}sec (that is {((end-start)/samples)*1000}ms on average!)")
