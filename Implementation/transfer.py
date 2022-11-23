import random
import numpy as np
import jax
import jax.numpy as npx
from jax_stuff import jax_dtype
import optax

from mlp import MyMLP
from autoencoder import Autoencoder


class TransferModel():
    def __init__(self, features, input_dim):
        if features[-1] != input_dim:
            raise ValueError(f"Output layer must have the same dimensionality as the input -- found {features[-1]} instead of {input_dim}")
        
        self.input_dim = input_dim
        self.model = MyMLP(features)

        # Randomly initialize parameters
        self.model_params = None
        self.__random_initialization()

    def __random_initialization(self):
        self.model_params = self.model.init(jax.random.PRNGKey(random.randint(0,100)), np.random.normal(size=self.input_dim))

    def params(self):
        return self.model_params
    
    def transform(self, X, params):
        return X + self.model.apply(params, X)

    def __call__(self, X):
        return self.transform(X, self.model_params)


class Transfer():
    def __init__(self, transfer_model, ae, C=1.):
        if not isinstance(transfer_model, TransferModel):
            raise TypeError(f"'transfer_model' must be an instance of 'TransferModel' not of {type(transfer_model)}")
        if not isinstance(ae, Autoencoder):
            raise TypeError(f"'ae_model' must be an instance of 'Autoencoder' not of {type(ae)}")
        
        self.transfer_model = transfer_model
        self.ae = ae
        self.ae_model = ae.ae_model
        self.C = C

    def __cost(self, X):
        regularization_prediction = lambda params: npx.sum(npx.abs(self.transfer_model.model.apply(params, X)))  # l1 regularization of prediction

        return lambda params: npx.mean(npx.square(self.transfer_model.transform(X, params) - self.ae_model(self.transfer_model.transform(X, params)))) + self.C * regularization_prediction(params)

    def fit(self, X, step_size=1e-3, n_iter=100, verbose=False):
        # Loss function
        loss = self.__cost(X)

        # Initialize gradient descent
        fval_fgrad = jax.value_and_grad(loss)

        # Line search for finding a good step size
        def compute_step_size(ss, f_val_grad, params, grads):
            if ss is not None:
                return ss
            else:
                step_size = [1. * 10**(-1. * i) for i in range(10)]
                f_vals = []
                for l in step_size:
                    myparams = jax.tree_map(lambda p, g: p - l * g, params, grads)

                    f_vals.append(f_val_grad(myparams)[0])

                return step_size[np.argmin(f_vals)]

        # Minimize loss by using gradient descent
        for _ in range(n_iter):
            # Compute gradients
            cur_loss, grads = fval_fgrad(self.transfer_model.model_params)

            # Deal with exploding gradients
            grads = jax.tree_map(lambda g: np.clip(g, -1., 1.), grads)  # Clip gradient

            # Perform one step of gradient descent
            # Custom implementation of gradient descent
            lr = compute_step_size(step_size, fval_fgrad, self.transfer_model.model_params, grads)
            self.transfer_model.model_params = jax.tree_map(lambda p, g: p - lr * g, self.transfer_model.model_params, grads)

            # Verbosity
            if verbose:
                print(cur_loss)
