import numpy as np
from sklearn.datasets import load_digits
from sklearn.model_selection import KFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

from autoencoder import Autoencoder, AutoencoderModel
from transfer import Transfer, TransferModel


def load_digits_data(batch1_classes, batch2_classes):
    X, y = load_digits(return_X_y=True)

    batch_1_idx = [y_i in batch1_classes for y_i in y]
    batch_2_idx = [y_i in batch2_classes for y_i in y]

    X_batch1, y_batch1 = X[batch_1_idx,:], y[batch_1_idx]
    X_batch2, y_batch2 = X[batch_2_idx,:], y[batch_2_idx]
    
    return X_batch1, y_batch1, X_batch2, y_batch2


def perturb_sample(x_img):
    n_dim = x_img.shape[0]
    n_dim_perturb = int(n_dim / 2.)  # Perturb the first half of pixels
    
    for i in range(n_dim_perturb):
        x_img[i] = 0.0  # Set pixel to zero (i.e. "turn it off")
        
    return x_img


if __name__ == "__main__":
    # Load and create data set
    X_batch1, y_batch1, X_batch2, y_batch2 = load_digits_data([0, 1, 2, 3, 4], [0, 1, 2, 3, 4])
    input_dim = X_batch1.shape[1]

    for i in range(X_batch2.shape[0]):  # Apply perturbation (concept drift) to second batch
        X_batch2[i,:] = perturb_sample(X_batch2[i,:])

    # Split the two sets/batches each further into a training and test set
    kf = KFold(n_splits=5)
    for train_index, test_index in kf.split(X_batch1):
        X_batch1_train, X_batch1_test, y_batch1_train, y_batch1_test = X_batch1[train_index, :], X_batch1[test_index,:], y_batch1[train_index], y_batch1[test_index]
        X_batch2_train, X_batch2_test, y_batch2_train, y_batch2_test = X_batch2[train_index, :], X_batch2[test_index,:], y_batch2[train_index], y_batch2[test_index]
        print(np.unique(y_batch1_train, return_counts=True))

        # Fit a classifier as a downstream task
        model = LogisticRegression()
        model.fit(X_batch1_train, y_batch1_train)

        print(f"Classifier score on clean test set: {model.score(X_batch1_test, y_batch1_test)}")
        print(f"Classifier score on faulty test set: {model.score(X_batch2_test, y_batch2_test)}")

        # Fit autoencoder for concept drift detection
        ae_model = AutoencoderModel(features=[32, 16, 32, input_dim], input_dim=input_dim)
        ae = Autoencoder(ae_model, C=.1)
        X_batch1_test_pred_0 = ae_model(X_batch1_test);score_0 = np.mean(np.square(X_batch1_test - X_batch1_test_pred_0))

        ae.fit(X_batch1_train, n_iter=1000, n_trials=5, step_size=1e-3, verbose=False)
        
        X_batch1_test_pred = ae_model(X_batch1_test)
        score = np.mean(np.square(X_batch1_test - X_batch1_test_pred))
        print(f"Autoencoder score before training the autoencoder: {score_0}")
        print(f"Autoencoder score after tranining the autoencoder: {score}")

        # Sanity check: Apply classifier to reconstructed samples!
        print(f"Classifier score on clean test set: {model.score(X_batch1_test, y_batch1_test)}")
        print(f"Classifier score on reconstructed clean test set: {model.score(X_batch1_test_pred, y_batch1_test)}")

        # Evaluate autoencoder on drifted data
        X_batch2_test_pred = ae_model(X_batch2_test)
        score_untransformed = np.mean(np.square(X_batch2_test - X_batch2_test_pred))
        print(f"Autoencoder score on fauly data: {score_untransformed}")
        print(f"Classifier score on faulty reconstructed test set: {model.score(X_batch2_test_pred, y_batch2_test)}")

        # Fit transfer function
        transfer_model = TransferModel(features=[input_dim], input_dim=input_dim)
        transfer = Transfer(transfer_model, ae, C=.00001)

        transfer.fit(X_batch2_train, n_iter=500, step_size=None, verbose=False)

        X_batch2_test_transformed = transfer_model(X_batch2_test)
        X_batch2_test_transformed_pred = ae_model(X_batch2_test_transformed)
        score_transformed = np.mean(np.square(X_batch2_test_transformed - X_batch2_test_transformed_pred))
        print(f"Autoencoder score on faulty AFTER transforming the data: {score_transformed}")

        # Evaluate downstream task
        print(f"Classifier score on faulty test set: {model.score(X_batch2_test, y_batch2_test)}")
        print(f"Classifier score on transformed faulty test set: {model.score(X_batch2_test_transformed, y_batch2_test)}")

        print(classification_report(y_batch1_test, model.predict(X_batch1_test)))
        print(classification_report(y_batch2_test, model.predict(X_batch2_test)))
        print(classification_report(y_batch2_test, model.predict(X_batch2_test_pred)))
        print(classification_report(y_batch2_test, model.predict(X_batch2_test_transformed)))

        print()
