import copy
import random
import numpy as np
import jax
import jax.numpy as npx
import optax

from mlp import MyMLP


class AutoencoderModel():
    def __init__(self, features, input_dim):
        if features[-1] != input_dim:
            raise ValueError(f"Output layer must have the same dimensionality as the input -- found {features[-1]} instead of {input_dim}")
        
        self.input_dim = input_dim
        self.model = MyMLP(features)

        # Randomly initialize parameters
        self.model_params = None
        self._random_initialization()

    def _random_initialization(self):
        self.model_params = self.model.init(jax.random.PRNGKey(random.randint(0,100)), np.random.normal(size=self.input_dim))

    def params(self):
        return self.model_params
    
    def transform(self, X, params):
        return self.model.apply(params, X)

    def __call__(self, X):
        return self.transform(X, self.model_params)


class Autoencoder():
    def __init__(self, ae_model, C=1.):
        if not isinstance(ae_model, AutoencoderModel):
            raise TypeError(f"'ae_model' must be an instance of 'AutoencoderModel' not of {type(ae_model)}")

        self.ae_model = ae_model
        self.C = C
        self.moments = []

    def __cost(self, X):
        regularization = lambda params: npx.sum(npx.array([npx.vdot(w, w) for w in jax.tree_util.tree_flatten(params)[0]]))   # l2
        return lambda params: npx.mean(npx.square(X - self.ae_model.transform(X, params))) + self.C * regularization(params)  # Mean-squared-error + regularization

    def fit(self, X, step_size=1e-3, n_iter=100, n_trials=2, verbose=False):
        # Train autoencoder
        best_loss = float("inf")
        best_ae_model = None
        
        for _ in range(n_trials):  # Use different initializations and use the best one!
            self.ae_model._random_initialization()
            
            # Loss function
            loss = self.__cost(X)

            # Initialize gradient descent
            learner = optax.chain(
                optax.clip(1.), # Gradient clipping to prevent exploding gradients
                optax.adam(step_size)
            )
            opt_state = learner.init(self.ae_model.model_params)
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
                cur_loss, grads = fval_fgrad(self.ae_model.model_params)

                # Deal with exploding gradients
                #grads = jax.tree_multimap(lambda g: np.clip(g, -1., 1.), grads)  # Clip gradient
                grads = jax.tree_map(lambda g: np.clip(g, -1., 1.), grads)  # Clip gradient

                # Perform one step of gradient descent
                updates, opt_state = learner.update(grads, opt_state)
                self.ae_model.model_params = optax.apply_updates(self.ae_model.model_params, updates)

                # Verbosity
                if verbose:
                    print(cur_loss)

            # Better parameters found?
            cur_params_loss = np.mean(np.square(self.ae_model(X) - X))
            if cur_params_loss < best_loss:
                best_loss = cur_params_loss
                best_ae_model = copy.deepcopy(self.ae_model)

        # Use best model
        self.ae_model = best_ae_model
