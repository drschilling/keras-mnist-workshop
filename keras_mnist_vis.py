# Authored by Raghavendra Kotikalapudi(ragha@outlook.com)

from vis.visualization import visualize_saliency
from vis.utils import utils
import numpy as np
from keras import activations
from matplotlib import pyplot as plt
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

# Metodo de visualizacao que mostra a variacao da classificacao
# dos digitos a partir de diferentes funcoes de ativacao
def keras_digits_vis(model, X_test, y_test):
    
    layer_idx = utils.find_layer_idx(model, 'preds')
    model.layers[layer_idx].activation = activations.linear
    model = utils.apply_modifications(model)

    for class_idx in np.arange(10):    
        indices = np.where(y_test[:, class_idx] == 1.)[0]
        idx = indices[0]

        f, ax = plt.subplots(1, 4)
        ax[0].imshow(X_test[idx][..., 0])
        
        for i, modifier in enumerate([None, 'guided', 'relu']):
            heatmap = visualize_saliency(model, layer_idx, filter_indices=class_idx, 
                                        seed_input=X_test[idx], backprop_modifier=modifier)
            if modifier is None:
                modifier = 'vanilla'
            ax[i+1].set_title(modifier)    
            ax[i+1].imshow(heatmap)
    plt.imshow(heatmap)
    plt.show()
