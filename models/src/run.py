import numpy as np
import tensorflow as tf
from gensim.models import Word2Vec
import tiktoken
from tensorflow.keras.models import load_model
from tensorflow.keras import Model
from tensorflow.keras import layers
from sklearn.metrics.pairwise import cosine_similarity
from chatGPT import GPT
import os
import random
import sys
sys.path.insert(0, "/mnt/data/dev/ML/lib/")
from SelectiveTransformation import SelectiveTransformation
from Transformers import Decoder

# === CONFIG ===
W2V_PATH = "../.compiled/cgpt/chatGPTv0_3e.model"
W2V_NUM = 3
KERAS_MODEL_PATH = f"../.compiled/cgpt/chatGPTv0_3@best.keras"
TEXT_END_TOKEN = "<endoftext>"  # Example, replace with your actual special token
SEQ_LENGTH = int(round(2048/6))  # Must match your model's context length
EMBEDDING_SIZE = 256  # Must match your training embedding size
LOOP_STEPS = 1
INPUT_STRING = ""
NUKLEUS_REQ = 0.3   # needs 10% to pass nukleus

# === LOAD WORD2VEC ===
print("[*] Loading Word2Vec model...")
w2v = []
for i in range(W2V_NUM):
    w2v.append(Word2Vec.load(f"../.compiled/cgpt/chatGPTv0_3@{i}e.model"))
w2v_vocab = list(w2v[0].wv.index_to_key)    # all w2v should have same vocab
w2v_vectors = np.hstack([w2v[i].wv[w2v_vocab] for i in range(W2V_NUM)])  # Shape: (vocab_size, embedding_dim)


# === CREATE TOKENIZER ===
print("[*] Creating tokenizer...")
base_tok = tiktoken.get_encoding("o200k_base")
tokenizer = tiktoken.Encoding(
    name="o200k_base",
    pat_str=base_tok._pat_str,
    mergeable_ranks=base_tok._mergeable_ranks,
    special_tokens={**base_tok._special_tokens, TEXT_END_TOKEN: 100264}
)

# === LOAD KERAS MODEL WITH CUSTOM LAYER ===
print("[*] Loading Keras model...")

# Example minimal Decoder placeholder (must match your training definition)
base = GPT()
model = load_model(KERAS_MODEL_PATH, custom_objects={"Decoder": Decoder})

# === MAP TOKENS TO EMBEDDINGS ===
def tokens_to_embeddings(tok_list):
    embeddings = []
    for tok in tok_list:
        if tok in w2v[0].wv:
            embeddings.append(np.hstack([w2v[i].wv[tok] for i in range(W2V_NUM)]))
        else:
            embeddings.append(np.zeros(EMBEDDING_SIZE))
    return np.array(embeddings)

    return random.choice(possible_tokens)
def embedding_to_token_top1(pred_emb):
    pred_emb = pred_emb.reshape(1, -1)
    sims = cosine_similarity(pred_emb, w2v_vectors)[0]
    return w2v_vocab[np.argmax(sims)]
def embedding_to_token_topP(pred_emb):
    pred_emb = pred_emb.reshape(1, -1)  # (1, embedding_dim)
    
    # Config
    embedding_sizes = [86, 85+86, 85+85+86][::-1]  # reversed already
    embedding_offsets = [0, 86, 85+86][::-1]  # start indices
    
    possible_tokens = set(w2v_vocab)  # start with full vocab
    
    # Process each slice
    for i, (start, end) in enumerate(zip(embedding_offsets, embedding_sizes)):
        current_embedding = pred_emb[:, start:end]  # slice
        # Compute similarities
        sims = cosine_similarity(
            current_embedding, 
            w2v[len(embedding_sizes)-1-i].wv[w2v_vocab][start:end]
        )[0]
        
        # Filter candidates
        mask = sims > NUKLEUS_REQ
        candidates = {tok for tok, keep in zip(w2v_vocab, mask) if keep}
        
        # Intersect with previous
        possible_tokens &= candidates
    
        if not possible_tokens:  # early stop if no overlap
            break
    
    return random.choice(list(possible_tokens)) if possible_tokens else "--"
# === PAD SEQUENCE ===
def pad_embeddings(embs):
    padded = np.zeros((SEQ_LENGTH, EMBEDDING_SIZE))
    length = min(len(embs), SEQ_LENGTH)
    padded[:length] = embs[:length]
    return padded

# === GENERATION LOOP ===
print("[*] Starting generation loop...")
while INPUT_STRING !="EXT":
    INPUT_STRING = input(">> ")
    tokens = tokenizer.encode(INPUT_STRING)
    tokens = [tokenizer.decode([token]) for token in tokens]
    LOOP_STEPS = 3
    os.system("clear")
    output_text = INPUT_STRING
    for i in range(LOOP_STEPS):
        embs = tokens_to_embeddings(tokens)
        padded_in = pad_embeddings(embs)
        pred_seq = model.predict(padded_in[np.newaxis, ...], verbose=0)[0]  # (seq_length, emb_dim)
        # pred_emb = pred_seq[len(tokens)]  # The *next* token embedding
        tokens = [embedding_to_token_top1(tok) for tok in pred_seq]
        next_tok = ""
        if next_tok == "<END>":
            break
        else:
            if isinstance(tokens[0], str):
                output_text += "".join(tokens)
            else:
                output_text += tokenizer.decode(tokens)
            linelen = len(output_text) + 3
            print(f">> {output_text}")
            sys.stdout.write("\033[H")
    if input(f">> {output_text}") == "\n":
        os.system("clear")
        print(">> ")
        continue
