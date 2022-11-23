import os
import numpy as np
import csv
from joblib import Parallel, delayed
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from autoencoder import Autoencoder, AutoencoderModel
from transfer import Transfer, TransferModel


data_path_in = "hanoi-data/"



def create_regression_problems_with_concept_drift(X, Y, y):
    X_data = []
    y_data = []

    n_features = X.shape[1]
    for i in range(n_features):
        inputs_idx = list(range(n_features));inputs_idx.remove(i)
        X_data.append(X[:,inputs_idx])
        y_data.append(Y[:,i])

    return X_data, y_data, y



if __name__ == "__main__":
    def process_file(file_in):
        try:
            output_data = []
            def log(info):
                output_data.append(info)

            # Load data
            data = np.load(os.path.join(data_path_in, file_in))
            X_final, Y_final, y_faulty = data["X_final"], data["Y_final"], data["y_faulty"]

            # Split data
            t_train_split = 3000
            n_samples_adaptation = 200  # Number of samples used for adapting the model to the fault (concept drift)

            faulty_times = np.where(y_faulty == 1)[0]
            test_t0, test_t1 = faulty_times[0], faulty_times[-1]  # Time period in the test set where a fault is present

            scaler = StandardScaler()
            input_dim = X_final.shape[1]

            X_all_train, X_all_test = X_final[:t_train_split,:], X_final[t_train_split:,:]
            Y_all_train, Y_all_test = Y_final[:t_train_split,:], Y_final[t_train_split:,:]

            X_before_fault = X_final[t_train_split:test_t0,:]
            Y_before_fault = Y_final[t_train_split:test_t0,:]
            X_before_fault = scaler.fit_transform(X_before_fault)

            # Data set when fault is present
            X_faulty = X_final[test_t0:test_t1,:]
            Y_faulty = Y_final[test_t0:test_t1,:]
            X_faulty = scaler.transform(X_faulty)

            # Data set for adaptation to the present fault
            X_adapt = X_final[test_t0:test_t0+n_samples_adaptation,:]
            X_adapt_eval = X_final[test_t0+n_samples_adaptation:test_t1,:]
            Y_adapt = Y_final[test_t0:test_t0+n_samples_adaptation,:]
            Y_adapt_eval = Y_final[test_t0+n_samples_adaptation:test_t1,:]

            X_adapt = scaler.transform(X_adapt)
            X_adapt_eval = scaler.transform(X_adapt_eval)

            # Data set after fault is over
            X_after_fault = X_final[test_t1:,:]
            Y_after_fault = X_final[test_t1:,:]

            X_after_fault = scaler.transform(X_after_fault)

            # Data set for adaptation after the fault
            X_adapt_2 = X_after_fault[:2*n_samples_adaptation,:]
            X_adapt_eval_2 = X_after_fault[2*n_samples_adaptation:,:]
            Y_adapt_2 = Y_after_fault[:2*n_samples_adaptation,:]
            Y_adapt_eval_2 = Y_after_fault[2*n_samples_adaptation:,:]

            X_adapt_2 = scaler.transform(X_adapt_2)
            X_adapt_eval_2 = scaler.transform(X_adapt_eval_2)

            # Fit downtream task
            X_all_train_ = scaler.transform(X_all_train)
            X_data, y_data, y_faulty = create_regression_problems_with_concept_drift(X_all_train_, Y_all_train, y_faulty)
            models = []
            for i in range(len(X_data)):
                X_train, y_train = X_data[i], y_data[i]

                model = Ridge()
                model.fit(X_train, y_train)
                models.append(model)

            # Evaluate down tream task
            X_data, y_data, y_faulty = create_regression_problems_with_concept_drift(X_before_fault, Y_before_fault, y_faulty)
            scores_before_fault = []
            before_fault_pred = []
            for i in range(len(X_data)):
                before_fault_pred.append((models[i].predict(X_data[i]), y_data[i]))
                scores_before_fault.append(models[i].score(X_data[i], y_data[i]))

            X_data, y_data, y_faulty = create_regression_problems_with_concept_drift(X_faulty, Y_faulty, y_faulty)
            scores_faulty = []
            for i in range(len(X_data)):
                scores_faulty.append(models[i].score(X_data[i], y_data[i]))

            # Fit autoencoder
            ae_model = AutoencoderModel(features=[10, input_dim], input_dim=input_dim)
            ae = Autoencoder(ae_model)
            X_before_fault_pred = ae_model(X_before_fault)
            score_before_fault_before_training = np.mean(np.square(X_before_fault - X_before_fault_pred))
            score_var_before_fault_before_training = np.var(np.square(X_before_fault - X_before_fault_pred))

            ae.fit(X_all_train, n_iter=800, n_trials=5, verbose=False)

            X_before_fault_pred = ae_model(X_before_fault)  # Evaluation
            before_fault_pred_diff=np.square(X_before_fault - X_before_fault_pred)
            score_before_fault = np.mean(before_fault_pred_diff)

            log(f"Autoencoder score on clean data BEFORE training: {score_before_fault_before_training}")
            log(f"Autoencoder score on clean data: {score_before_fault}")

            # Use autoencoder as a concept drift detector
            X_faulty_pred = ae_model(X_faulty)

            faulty_pred_diff=np.square(X_faulty - X_faulty_pred)
            score_fault = np.mean(faulty_pred_diff)
            log(f"Autoencoder score on faulty data: {score_fault}")

            # Fit transfer function
            transfer_model = TransferModel(features=[input_dim], input_dim=input_dim)
            transfer = Transfer(transfer_model, ae, C=0.001)

            transfer.fit(X_adapt, n_iter=500, step_size=None, verbose=False)

            X_adapt_transformed = transfer_model(X_adapt_eval)  # Evaluate transfer function

            X_adapt_eval_pred = ae_model(X_adapt_eval)
            adapt_eval_diff=np.square(X_adapt_eval - X_adapt_eval_pred)
            score_fault = np.mean(adapt_eval_diff)

            X_adapt_transformed_eval_pred = ae_model(X_adapt_transformed)
            adapt_transformed_eval_diff=np.square(X_adapt_transformed - X_adapt_transformed_eval_pred)
            score_transformed_fault = np.mean(X_adapt_eval_pred)

            log(f"Autoencoder on untransformed faulty data: {score_fault}")
            log(f"Autoencoder on transformed faulty data: {score_transformed_fault}")

            # Evaluate downtream task
            X_data, y_data, y_faulty = create_regression_problems_with_concept_drift(X_adapt_eval_pred, Y_adapt_eval, y_faulty) # Baseline: Reconstructued sample from autoencoder
            scores_reconstructed_faulty = []
            adapt_eval_ae_pred = []
            for i in range(len(X_data)):
                adapt_eval_ae_pred.append((models[i].predict(X_data[i]), y_data[i]))
                scores_reconstructed_faulty.append(models[i].score(X_data[i], y_data[i]))

            X_data, y_data, y_faulty = create_regression_problems_with_concept_drift(X_adapt_eval, Y_adapt_eval, y_faulty)
            scores_faulty = []
            adapt_eval_pred = []
            for i in range(len(X_data)):
                adapt_eval_pred.append((models[i].predict(X_data[i]), y_data[i]))
                scores_faulty.append(models[i].score(X_data[i], y_data[i]))

            X_data, y_data, y_faulty = create_regression_problems_with_concept_drift(X_adapt_transformed, Y_adapt_eval, y_faulty)
            scores_transformed_faulty = []
            adapt_eval_tansformed_pred = []
            for i in range(len(X_data)):
                adapt_eval_tansformed_pred.append((models[i].predict(X_data[i]), y_data[i]))
                scores_transformed_faulty.append(models[i].score(X_data[i], y_data[i]))

            # Store results
            np.savez(os.path.join(data_path_in, file_in.replace(".npz", ".npz_results")), before_fault_pred=before_fault_pred, adapt_eval_ae_pred=adapt_eval_ae_pred, adapt_eval_pred=adapt_eval_pred, adapt_eval_tansformed_pred=adapt_eval_tansformed_pred)

            file_out_txt = file_in.replace(".npz", ".txt")
            with open(os.path.join(data_path_in, file_out_txt), "w") as f_out:
                f_out.write("\n".join(output_data))
            
            file_out_csv = file_in.replace(".npz", ".csv")
            with open(os.path.join(data_path_in, file_out_csv), "w") as csvfile:  
                csvwriter = csv.writer(csvfile)
                
                for i in range(len(scores_before_fault)):
                    csvwriter.writerow([scores_before_fault[i], scores_faulty[i], scores_reconstructed_faulty[i], scores_transformed_faulty[i]])
        except Exception as ex:
            print(ex)

    # Enumerate all files
    sub_folder_in = "hanoi_faultysensor/"

    files_in = list(filter(lambda f: f.endswith(".npz"), os.listdir(os.path.join(data_path_in, sub_folder_in))))
    files_in = [os.path.join(sub_folder_in, file_in) for file_in in files_in]

    Parallel(n_jobs=-2)(delayed(process_file)(file_in) for file_in in files_in)
