from keras import layers, Model, RematScope
import config
import sys
sys.path.insert(0, "/mnt/data/dev/ML/lib/")
from SelectiveTransformation import SelectiveTransformation
from Transformers import Decoder
print(config._parent_dir)

class GPT():
    def __init__(self):
        self.version = "v0_3"
        self.parent_dir = "cgpt"
        self.vocab_size = 8000  # useless
        
        self.embedding_size = 256
        self.embedding_num = 3
        import math
        self.sub_embedding_size = int(math.floor(self.embedding_size/self.embedding_num))
        print(f"{"\033[1;36m"}each embedding will get {self.sub_embedding_size} units where schematic gets {self.embedding_size%self.embedding_num} more{"\033[0m"}")
        del math
        self.schematic_window = 32
        self.grammar_window = 5

        self.context_length = int(round(2048/6))
        self.embeddings = [None]
        self.encoder = None
        self.TextEndToken = "<|endoftext|>"
        self.wikis = ["Informatik", "Eingebettetes_System", "AT&T", "Bell_Laboratories", "Python_(Programmiersprache)", "C_(Programmiersprache)", "C++", "Assemblersprache", "Prozessor", "Grafikprozessor", "KI-Beschleuniger", "Künstliche_Intelligenz", "Maschinelles_Lernen", "Interpreter", "Compiler", "Just-in-time-Kompilierung", "Vulkan_(API)", "Linux", "Syntaxbaum", "For-Schleife", "Schleife_(Programmierung)", "Bytecode", "Variable_(Programmierung)", "Backpropagation", "Blender_(Software)", "Physik", "Grafik-Engine", "Spiel-Engine", "ChatGPT", "Convolutional_Neural_Network", "Deep_Learning", "Starrer_Körper", "Flynnsche_Klassifikation", "X86-Architektur", "Nvidia", "Intel", "AMD", "ARM-Architektur", "Transmission_Control_Protocol/Internet_Protocol", "User_Datagram_Protocol", "File_Transfer_Protocol", "Ping_(Datenübertragung)", "Server", "Client", "Artificial_General_Intelligence", "KI-Sicherheit", "Soziale_Medien", "Objektorientierte_Programmierung", "Funktionale_Programmierung", "Funktion_(Programmierung)", "Arithmetisch-logische_Einheit", "Bitweiser_Operator", "Logikgatter", "Klasse_(Objektorientierung)", "Java_(Programmiersprache)", "Java_Virtual_Machine", "C-Sharp", "Google", "Wikipedia", "Programmierparadigma", "Superintelligenz", "Technologische Singularität"]
        ### wikis:
        # general purpose:
        # self.wikis += [wiki for wiki in ["Heinrich_der_Löwe", "Chronik_des_russischen_Überfalls_auf_die_Ukraine_ab_2025", "Kernwaffe", "The_Beatles", "Iren", "Toaster", "Künstliche_Intelligenz", "Universum", "Musik", "Mensch", "Vereinigtes_Königreich", "Wikipedia", "Familie", "Freundschaft", "Alltag", "Kommunikation", "Sprache", "Wohnung", "Stadt", "Dorf", "Straße", "Tag", "Woche", "Zeit", "Frühstück", "Essen", "Trinken", "Küche", "Obst", "Gemüse", "Spiel", "Sport", "Musik", "Lesen", "Glück", "Gedanke", "Angst", "Lachen", "Wasser", "Luft", "Licht", "Kleidung", "Gegenstand", "Liste_deutscher_Redewendungen", "Python_(Programmiersprache)", "C_(Programmiersprache)", "Terminator_(Film)", "Erde", "Bestärkendes_Lernen", "OpenAI", "Kunst"] if wiki not in self.wikis]
        self.model = None
    
    def subdivide(self, arr):
        seq_length, embedding_dim = arr.shape
        chunks = []

        # Calculate how many full chunks we have
        for i in range(0, seq_length, self.context_length):
            chunk = arr[i:i+self.context_length]

            if chunk.shape[0] < self.context_length:
                # Padding needed
                pad_size = self.context_length - chunk.shape[0]
                pad = np.zeros((pad_size, self.embedding_size), dtype=arr.dtype)

                if self.TextEndToken is not None:
                    # Insert <END> token at first padding slot
                    pad[0] = self.embedding.wv[self.TextEndToken]

                chunk = np.vstack([chunk, pad])

            chunks.append(chunk)

        return chunks
    def wikiget(self, title: str):
        import requests
        response = requests.get(
            'https://de.wikipedia.org/w/api.php',
            params={
                'action': 'query',
                'format': 'json',
                'titles': title,
                'prop': 'extracts',
                'explaintext': True,
            }
        )
        print(f"{'\033[1;36m'}wiki fetch passed with: {'\033[1;30m' if response.status_code==200 else '\033[1;31m'}{response.status_code}{'\033[1;36m'} for {title}{'\033[0m'}")
        if response.status_code==429:
            import time
            time.sleep(5)
            print(f"{'\033[1;36m'}awaited timeout{'\033[0m'}")
            return self.wikiget(title)
        page = next(iter(response.json()['query']['pages'].values()))
        return page['extract']
    def dataset(self):
        dataset = []
        for wiki in self.wikis:
            try:
                dataset.append(self.wikiget(wiki))
            except Exception as E:
                print(f"{'\033[1;31m'}Warning: could not get {wiki} due to {E}{'\033[0m'}")
                continue
        return dataset
    def __init_embedding__(self, dataset=None, overwrite: bool=False):
        from gensim.models import Word2Vec
        import tiktoken
        from os import cpu_count
        self.tokenizer = tiktoken.get_encoding("o200k_base")
        self.tokenizer = tiktoken.Encoding(
            # If you're changing the set of special tokens, make sure to use a different name
            # It should be clear from the name what behaviour to expect.
            name="o200k_base",
            pat_str=self.tokenizer._pat_str,
            mergeable_ranks=self.tokenizer._mergeable_ranks,
            special_tokens={
                **self.tokenizer._special_tokens,
               self.TextEndToken: 100264,
            }
        )
        sentences_raw = dataset if dataset is not None else self.dataset()
        sentences_tok = []
        for sentence in sentences_raw:
            token_ids = self.tokenizer.encode(sentence, allowed_special="all")
            sentences_tok.append([self.tokenizer.decode([tid]) for tid in token_ids]+[self.TextEndToken]) 
        
        if self.embeddings[0]!=None:
            print(f"{'\033[1;33m'}Warning: overwriting embedding{'\033[0m'}")
            if overwrite:
                pass
            else:
                return sentences_tok
        import math
        for i in range(self.embedding_num):
            print(f"creating one with window={int(((self.embedding_num-i-1)/(self.embedding_num-1))*self.schematic_window)+int((i/(self.embedding_num-1))*self.grammar_window)}")
            self.embeddings.append(Word2Vec(sentences_tok, vector_size=int(math.floor(self.embedding_size/self.embedding_num))+[int(self.embedding_size%self.embedding_num) if i==0 else 0][0], window=int(((self.embedding_num-i-1)/(self.embedding_num-1))*self.schematic_window)+int((i/(self.embedding_num-1))*self.grammar_window), min_count=1, workers=cpu_count()))       #to be applied OUTSIDE of class. not doin this man...
        self.embeddings.pop(0)
        print(f"{'\033[1;36m'}Finished training word embeddings{'\033[0m'}")
        return sentences_tok
    def _save(self, dataset=None):
        from zipfile import ZipFile
        archiveName = f"{config._parent_dir}{self.parent_dir}/chatGPT{self.version}a.zip"
        try:
            # open zip archive if it exists
            archive = ZipFile(archiveName, "r")
        except:
            # create if it doesn't exit
            from os import makedirs
            from shutil import rmtree
            makedirs(f"{config._parent_dir}{self.parent_dir}/chatGPT{self.version}d/")
            archive = ZipFile(archiveName, "w")
            for wiki in range(len(self.wikis)):
                wikiName = ''.join([char if char!='/' else '%' for char in self.wikis[wiki]])
                try:
                    # if file exists don't bother fetching wiki and rewriting
                    # something went wrong since dir exists but isnt' archived...
                    open(f"{config._parent_dir}{self.parent_dir}/chatGPT{self.version}d/{wikiName}.txt", "x")
                    print(f"{"\033[1;30m"}Creating and writing to file {wikiName}.txt and writing to archive{"\033[0m"}")
                    open(f"{config._parent_dir}{self.parent_dir}/chatGPT{self.version}d/{wikiName}.txt", "w").write(self.wikiget(self.wikis[wiki]))
                except:
                    print(f"{"\033[1;37m"}File {wikiName}.txt exists in uncompressed archive; skipping{"\033[0m"}")
                archive.write(f"{config._parent_dir}{self.parent_dir}/chatGPT{self.version}d/{wikiName}.txt")
            archive.close()
            rmtree(f"{config._parent_dir}{self.parent_dir}/chatGPT{self.version}d/")
            archive = ZipFile(archiveName, "r")
        for embedding in range(len(self.embeddings)):
            self.embeddings[embedding].save(f"{config._parent_dir}{self.parent_dir}/chatGPT{self.version}@{embedding}e.model")
        del ZipFile
        try:
            del makedirs
            del rmdir
        except:
            pass
    def _load(self):
        from gensim.models import Word2Vec
        from zipfile import ZipFile
        archive = None
        try:
            archive = ZipFile(f"{config._parent_dir}{self.parent_dir}/chatGPT{self.version}a.zip", "r")
            print(f"{"\033[1;30m"}loaded archive{"\033[0m"}")
        except:
            print(f"{"\033[1;32m"}could not load zipfile{"\033[0m"}")
        archived = [fileName.split("/")[-1][0:-4].replace("%", "/") for fileName in archive.namelist()] if archive is not None else []
        dataset = [archive.open(f"{config._parent_dir}{self.parent_dir}/chatGPT{self.version}d/{file.replace("/", "%")}.txt", "r").read().decode("utf-8") for file in archived]
        for wiki in [wiki for wiki in self.wikis if wiki not in archived]:
            dataset.append(self.wikiget(wiki))
        try:
            for i in range(self.embedding_num):
                self.embeddings.append(Word2Vec.load(f"{config._parent_dir}{self.parent_dir}/chatGPT{self.version}@{i}e.model"))
            print(f"{"\033[1;30m"}loaded embedding{"\033[0m"}")
        except:
            print(f"{"\033[1;32m"}could not load embedding{"\033[0m"}")
        return dataset
    def GPT(self):
        a = 2
        a = 4
        inputs = layers.Input((self.context_length, self.embedding_size))
        pos_enc = keras_hub.layers.SinePositionEncoding(max_wavelength=self.context_length)(inputs)
        pos_emb = keras_hub.layers.PositionEmbedding(self.context_length)(inputs)
        encoding = layers.Add()([inputs, pos_emb])
        trans1 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False, dropout_rate=0.15, attention_dropout=0.2)(encoding)
        trans2 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False, dropout_rate=0.15, attention_dropout=0.2)(trans1)
        trans3 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False, dropout_rate=0.15, attention_dropout=0.2)(trans2)
        trans4 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False, dropout_rate=0.15, attention_dropout=0.2)(trans3)
        trans5 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False, dropout_rate=0.15, attention_dropout=0.2)(trans4)
        trans6 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False, dropout_rate=0.15, attention_dropout=0.2)(trans5)
        trans7 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False, dropout_rate=0.15, attention_dropout=0.2)(trans6)
        trans8 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False, dropout_rate=0.15, attention_dropout=0.2)(trans7)
        # trans9 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans8)
        # trans10 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans9)
        # trans11 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans10)
        # trans12 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans11)
        # trans13 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans12)
        # trans14 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans13)
        # trans15 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans14)
        # trans16 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans15)
        # trans17 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans16)
        # trans18 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans17)
        # trans19 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans18)
        # trans20 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans19)
        # trans21 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans20)
        # trans22 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans21)
        # trans23 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans22)
        # trans24 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans23)
        # trans25 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans24)
        # trans26 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans25)
        # trans27 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans26)
        # trans28 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans27)
        # trans29 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans28)
        # trans30 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans29)
        # trans31 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans30)
        # trans32 = Decoder(seq_length=self.context_length, embedding_size=self.embedding_size, heads=8, hidden_size=a, activation="swish", use_flashattention=False)(trans31)
        out = layers.Dense(units=self.embedding_size, activation=None)(trans8)
        self.model = Model(inputs=inputs, outputs=out)
        return self.model
if __name__=="__main__":
    import keras
    import numpy as np
    import keras_hub
    import random
    import os
    import datetime
    import matplotlib.pyplot as plt
    myGPT = GPT()
    model = myGPT.GPT()
   
    optimizerF = keras.optimizers.Adam(
        learning_rate=1e-8,     # high enough to move fast, small data means overfit is fine
        beta_1=0.9,             # good default for momentum
        beta_2=0.98,            # slightly lower than 0.999 → adapts to small datasets faster
        epsilon=1e-8,           # standard; keeps stability
    )
    start_lrF = 1e-9
    start_lr = 1e-10
    optimizer = keras.optimizers.AdamW(
        learning_rate=start_lr,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-8,
        weight_decay=0.01,
    )
    target_lrF = 7.5e-9
    target_lr = 1e-9
    model.compile(
        optimizer=optimizerF,
        loss=keras.losses.CosineSimilarity(),
        loss_weights=None,
        metrics=[keras.metrics.CosineSimilarity(axis=-1)],
        weighted_metrics=None,
        run_eagerly=False,
        steps_per_execution=1,
        jit_compile="auto",
    )
    
    dataset = myGPT._load()
    dataset = myGPT.__init_embedding__(dataset = dataset, overwrite=False)
    dataset_sen = dataset
    myGPT._save()
    dataset = [np.array([np.concatenate((myGPT.embeddings[0].wv[word], myGPT.embeddings[1].wv[word], myGPT.embeddings[2].wv[word])) for word in wiki if word in myGPT.embeddings[0].wv]) for wiki in dataset]
    episodes = 10000
    velocity = 0.9
    warmupsteps = int(round((1-velocity)*episodes))
    losses = []
    for episode in range(episodes+1):
        if episode<warmupsteps:
            optimizer.learning_rate -= (1/(warmupsteps)) * (start_lrF-target_lrF)
        x = []
        y = []
        val_x = []
        val_y = []
        c = 0
        for i in range(len(dataset)):
            point = dataset[i]
            if len(point)<2*myGPT.context_length:
                print(f"{"\033[1;35m"}Warning: Wiki Article too short: {"\033[1;36m"}{myGPT.wikis[i]} {"\033[1;30"}{myGPT.context_length}:{len(point)}{"\033[0m"}")
                continue
            x_max_index = point.shape[0] - 2*myGPT.context_length
            index = random.randint(0, x_max_index)
            queue_length = random.randint(1, myGPT.context_length)
            
            x.append(np.zeros((myGPT.context_length, myGPT.embedding_size)))
            x[i][0:queue_length] = point[index:index+queue_length]
            y.append(point[index+queue_length:index+queue_length+myGPT.context_length])
            val_index = random.randint(0, x_max_index)
            val_queue_length = random.randint(1, myGPT.context_length)
            
            val_x.append(np.zeros((myGPT.context_length, myGPT.embedding_size)))
            val_x[i][0:val_queue_length] = point[val_index:val_index+val_queue_length]
            val_y.append(point[val_index+val_queue_length:val_index+val_queue_length+myGPT.context_length])
        history = model.fit(np.array(x), np.array(y), batch_size=4, epochs=1, validation_data=[np.array(val_x), np.array(val_y)])
        losses += history.history["loss"]
        print(f"{'\033[1;36m'}finished {'\033[1;30m'}{episode}{'\033[1;36m'} out of {'\033[1;33m'}{episodes} {'\033[1;36m'}episodes{'\033[0m'}")
        if int(round(np.argmin(losses)))==episode:
            print(f"{"\033[1;30m"}Episode: {episode}: Best Model at {int(round(np.argmin(losses)))} with {losses[episode]}; saving{"\033[0m"}")
            model.save(f"{config._parent_dir}{myGPT.parent_dir}/chatGPT{myGPT.version}@best.keras")
        if episode%50==0:
            print(f"{"\033[1;33m"}Episode {episode}: Backup{"\033[0m"}")
            try:
                model.save(f"{config._parent_dir}{myGPT.parent_dir}/chatGPT{myGPT.version}@{episode}.keras")
            except:
                print(f"{"\033[1;32m"}ERROR: No space left on device{"\033[0m"}\n")
    model.save(f"{config._parent_dir}{myGPT.parent_dir}/chatGPT{myGPT.version}.keras")
    plt.plot([x for x in range(len(losses))], losses)
    plt.savefig(f"{config._parent_dir}{myGPT.parent_dir}/chatGPT{myGPT.version}m.png")
