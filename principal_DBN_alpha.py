from principal_RBM_alpha import RBM
import numpy as np
import matplotlib.pyplot as plt
import scipy.io

class DBN:

    def __init__(self, layer_sizes):
        self.rbms = [
            RBM(layer_sizes[i], layer_sizes[i + 1]) for i in range(len(layer_sizes) - 1)
        ]  

    def train_DBN(self, data, epochs, learning_rate, batch_size, dnn = False):
        errors = []
        input_data = data
        if dnn:
            for i, rbm in enumerate(self.rbms[:-1]):
                print(f"PRETraining RBM for DNN {i+1}/{len(self.rbms)}")
                rbm.train_RBM(input_data, epochs, learning_rate, batch_size)
                input_data = rbm.entree_sortie_RBM(input_data)
        else:
            for i, rbm in enumerate(self.rbms):
                print(f"Training RBM {i+1}/{len(self.rbms)}")
                error = rbm.train_RBM(input_data, epochs, learning_rate, batch_size)
                errors.append(error)
                input_data = self.rbms[i].entree_sortie_RBM(input_data)
            return errors

    def generer_image_DBN(self, iter_gibbs, nb_images_a_generer):
        if not self.rbms:
            raise ValueError("No RBMs in the DBN. Make sure to train the model first.")

        images = []
        for _ in range(nb_images_a_generer):
            visible = (np.random.rand(self.rbms[-1].a.shape[0]) < 0.5).astype(float)

            for _ in range(iter_gibbs):
                prob1 = self.rbms[-1].entree_sortie_RBM(visible)
                hidden = (np.random.rand(self.rbms[-1].b.shape[0]) < prob1).astype(float)
                prob2 = self.rbms[-1].sortie_entree_RBM(hidden)
                visible = (np.random.rand(self.rbms[-1].a.shape[0]) < prob2).astype(float)

            for i in range(len(self.rbms) - 2, -1, -1):
                hidden_probs = self.rbms[i].entree_sortie_RBM(visible)
                hidden_state = (
                    np.random.rand(self.rbms[i].b.shape[0]) < hidden_probs
                ).astype(float)
                visible_probs = self.rbms[i].sortie_entree_RBM(hidden_state)
                visible = (
                    np.random.rand(self.rbms[i].a.shape[0]) < visible_probs
                ).astype(float)

            images.append(visible)

        return np.array(images)