import numpy as np
import matplotlib.pyplot as plt
import scipy.io


def lire_alpha_digit(ds, elements):
    X = []
    for elem in elements:
        for i in ds[elem]:
            X.append(i.flatten())
    return np.array(X)

class RBM:      
    def __init__(self, n_visible, n_hidden):
        self.a = np.zeros(n_visible)
        self.b = np.zeros(n_hidden) 
        self.w = np.random.normal(loc=0.0, scale=0.1, size=(n_visible, n_hidden))

    def sigmoid(self, x):
        return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x))) 

    def entree_sortie_RBM(self, data):
        """Compute the hidden layer activation probabilities."""
        return self.sigmoid(np.dot(data, self.w) + self.b)

    def sortie_entree_RBM(self, hidden):
        """Compute the visible layer activation probabilities."""
        return self.sigmoid(np.dot(hidden, self.w.T) + self.a)

    def clip_gradients(self, gradients, threshold=1.0):
        return np.clip(gradients, -threshold, threshold)

    def reconstruction_error(self, data):
        recon = self.sortie_entree_RBM(self.entree_sortie_RBM(data))
        return np.mean(np.sum((data - recon) ** 2, axis=1))

    def train_RBM(self, data, epochs, learning_rate, batch_size):
        """Train an RBM using Contrastive Divergence-1."""
        n_samples = data.shape[0]
        errors = []
        initial_learning_rate = learning_rate
        decay_rate = 0.95  # Decay rate per epoch

        for epoch in range(epochs):
            np.random.shuffle(data)
            for i in range(0, n_samples, batch_size):
                batch = data[i : i + batch_size]

                # Positive phase
                h_pos_prob = self.entree_sortie_RBM(batch)
                pos_prod = np.dot(batch.T, h_pos_prob)
                h_sample = (
                    np.random.rand(batch.shape[0], self.b.shape[0]) < h_pos_prob
                ).astype(float)

                # Negative phase
                v_neg_prob = self.sortie_entree_RBM(h_sample)
                v_sample = (
                    np.random.rand(batch.shape[0], self.a.shape[0]) < v_neg_prob
                ).astype(float)
                h_neg_prob = self.entree_sortie_RBM(v_sample)
                negative_prod = np.dot(v_sample.T, h_neg_prob)

                # Update weights and biases
                grad_w = (pos_prod - negative_prod) / batch.shape[0]
                grad_a = np.mean(batch - v_sample, axis=0)
                grad_b = np.mean(h_pos_prob - h_neg_prob, axis=0)

                grad_w = self.clip_gradients(grad_w)
                grad_a = self.clip_gradients(grad_a)
                grad_b = self.clip_gradients(grad_b)

                self.w += learning_rate * grad_w
                self.a += learning_rate * grad_a
                self.b += learning_rate * grad_b

            # Decay learning rate
            learning_rate = initial_learning_rate * (decay_rate**epoch)

            # Reconstruction error
            error = self.reconstruction_error(data)
            print(f"Epoch {epoch+1}/{epochs}, Reconstruction Error: {error}")
            errors.append(error)
        return errors

    def generer_image_RBM(self, nb_iterations_gibbs, nb_images_a_generer):
        num_visible = self.a.shape[0]
        num_hidden = self.b.shape[0]

        generated_images = []

        for _ in range(nb_images_a_generer):
            visible_state = (np.random.rand(num_visible) < 0.5).astype(float)

            for _ in range(nb_iterations_gibbs):
                hidden_probs = self.sigmoid(np.dot(visible_state, self.w) + self.b)
                hidden_state = (np.random.rand(num_hidden) < hidden_probs).astype(float)

                visible_probs = self.sigmoid(np.dot(hidden_state, self.w.T) + self.a)
                visible_state = (np.random.rand(num_visible) < visible_probs).astype(
                    float
                )

            generated_images.append(visible_state)

        return generated_images
