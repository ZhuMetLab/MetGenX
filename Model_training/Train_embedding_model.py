"""
# File       : Train_embedding_model.py
# Time       : 2024/10/21 14:09
# Author     : Hongmiao Wang
# version    : python 3.10
# Description: 
"""
import gensim
from gensim.models.callbacks import CallbackAny2Vec
import os
from TemplateSearch.Embedding import GenerateSpec2vec
import pandas as pd

def train_gensim_model(Training_mz, model_save_dir):
    class LossLogger(CallbackAny2Vec):
        '''Output loss at each epoch'''

        def __init__(self):
            self.epoch = 0
            self.losses = []

        def on_epoch_begin(self, model):
            print(f'Epoch: {self.epoch}', end='\t')

        def on_epoch_end(self, model):
            loss = model.get_latest_training_loss()
            if self.epoch == 0:
                print('Loss after epoch {}: {}'.format(self.epoch, loss))
                self.min_loss = loss
                self.losses.append(loss)
            else:
                epoch_loss = loss - self.loss_previous_step
                self.losses.append(epoch_loss)
                print('Loss after epoch {}: {}'.format(self.epoch, epoch_loss))
                if self.min_loss > epoch_loss:
                    self.min_loss = epoch_loss
                    model.save(os.path.join(model_save_dir, 'SpecEmbed_model'.format(
                        self.epoch)))
                    print("Model saved SpecEmbed_model".format(
                        self.epoch))
            self.epoch += 1
            self.loss_previous_step = loss

    loss_logger = LossLogger()
    model = gensim.models.Word2Vec(Training_mz, sg=0, negative=5, vector_size=512, window=600, min_count=10,
                                   workers=4,
                                   compute_loss=True, alpha=0.025, min_alpha=0.0125, epochs=50, seed=42,
                                   callbacks=[loss_logger])
    return model


if __name__ == '__main__':
    project_dir = "./data/MassSpecGym"
    metadata = pd.read_csv(os.path.join(project_dir, "MassSpecGym.tsv"), sep="\t")
    Training_mz = []
    from tqdm import tqdm
    for step, row in tqdm(metadata.iterrows()):
        mz = [float(m) for m in row["mzs"].split(",")]
        intensity = [float(m) for m in row["intensities"].split(",")]
        precursor_mz = row["precursor_mz"]
        spec_token, spec_intensity = GenerateSpec2vec(mz, intensity,
                                                      round(precursor_mz, 2)
                                                      , TopN=100, TopN_mz=None, min_int=0.01, NL_range=[0.5, 200],
                                                      min_frag=1)
        Training_mz.append(spec_token)
    if not os.path.exists(os.path.join(project_dir, "weights", "word2vec")):
        os.makedirs(os.path.join(project_dir, "weights", "word2vec"))
    model = train_gensim_model(Training_mz, model_save_dir=os.path.join(project_dir, "weights/word2vec/"))