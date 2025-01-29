import numpy as np
import copy
from principal_DBN_alpha import DBN


class DNN:
    def __init__(self, tailles_couches):
        self.dbn = DBN(tailles_couches)

    def pretrain_DNN(self, epoques, learning_rate, batch_size, donnees_entree):
        self.dbn.train_DBN(copy.deepcopy(donnees_entree), epoques, learning_rate, batch_size, dnn=True)

    def calcul_softmax(self, sortie):
        shiftx = sortie - np.max(sortie, axis=1, keepdims=True)  # Numerical stability
        softmax = np.exp(shiftx) / np.sum(np.exp(shiftx), axis=1, keepdims=True)
        return softmax

    def entree_sortie_reseau(self, donnees_entree):
        sorties_couches = []
        sorties_couches.append(donnees_entree)
        entree_actuelle = copy.deepcopy(donnees_entree)

        for rbm in self.dbn.rbms:
            entree_actuelle = rbm.entree_sortie_RBM(entree_actuelle)
            sorties_couches.append(entree_actuelle)

        sorties_couches[-1] = np.log(sorties_couches[-1] / (1 - sorties_couches[-1]))
        proba_clf = self.calcul_softmax(sorties_couches[-1])

        return sorties_couches, proba_clf

    def retropropagation(self, epoques, taux_apprentissage, taille_lot, donnees_entree, etiquettes):
        nombre_echantillons = len(donnees_entree)

        for epoque in range(epoques):
            entropie_totale = 0.0

            for i in range(0, nombre_echantillons, taille_lot):
                lot_entree = donnees_entree[i : min(i + taille_lot, nombre_echantillons)]
                lot_etiquettes = etiquettes[i : min(i + taille_lot, nombre_echantillons)]
                n_lot = lot_entree.shape[0]

                sorties_couches, proba_clf = self.entree_sortie_reseau(lot_entree)

                delta = proba_clf - lot_etiquettes

                dbn_copy = copy.deepcopy(self.dbn)
                for couche in range(len(dbn_copy.rbms) - 1, -1, -1):
                    grad_w = np.dot(sorties_couches[couche].T, delta) / n_lot
                    grad_b = np.mean(delta, axis=0)
                    dbn_copy.rbms[couche].w -= taux_apprentissage * grad_w
                    dbn_copy.rbms[couche].b -= taux_apprentissage * grad_b
                    delta = (
                        np.dot(delta, self.dbn.rbms[couche].w.T)
                        * sorties_couches[couche]
                        * (1 - sorties_couches[couche])
                    )

                self.dbn = dbn_copy

            _, Y_est = self.entree_sortie_reseau(donnees_entree)
            entropie_totale += -np.sum(etiquettes * np.log(Y_est + 1e-10))
            entropie_moyenne = entropie_totale / nombre_echantillons
            print(f"Epoch {epoque + 1}/{epoques}, Cross Entropy: {entropie_moyenne}")

        return entropie_moyenne

    def test_DNN(self, donnees_test, etiquettes_vraies):
        _, Y_est = self.entree_sortie_reseau(donnees_test)

        taux_erreur = np.mean(np.argmax(Y_est, axis=1) != np.argmax(etiquettes_vraies, axis=1))

        return taux_erreur
