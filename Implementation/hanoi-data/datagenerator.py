import os
import numpy as np
import pandas as pd

from LeakDbScenario import get_scenarios_with_without_leakages, Scenario


def global_preprocessing(X, y_labels, sensors_idx=None, time_start=100, time_win_len=3):
    X_final = []
    Y_final = []
    y_fault = []
    
    if sensors_idx is None:
        sensors_idx = list(range(X.shape[1]))
    
    # Use a sliding time window to construct a labeled data set
    t_index = time_start
    time_points = range(len(y_labels))
    i = 0
    while t_index < len(time_points) - time_win_len:
        # Grab time window from data stream
        x = X[t_index:t_index+time_win_len-1, sensors_idx]

        #######################
        # Feature engineering #
        #######################
        x = np.mean(x,axis=0)  # "Stupid" feature
        X_final.append(x)

        Y_final.append([X[t_index + time_win_len-1, n] for n in sensors_idx])

        y_fault.append(y_labels[t_index + time_win_len-1])

        t_index += 1  # Note: Overlapping time windows
        i += 1

    X_final = np.array(X_final)
    Y_final = np.array(Y_final)
    y_fault = np.array(y_fault)

    return X_final, Y_final, y_fault



def prepare_clean_hanoi_data(path_to_data, dir_out="hanoi_clean/"):
    scenarios_without_leaks, _ = get_scenarios_with_without_leakages(path_to_data)

    i = 0
    for s_id, i in zip(scenarios_without_leaks, range(len(scenarios_without_leaks))):
        print(f"{i}: {s_id}")

        scenario = Scenario(s_id, path_to_data)
        pressure_nodes = scenario.node_ids

        y = scenario.labels
        X = np.vstack([scenario.pressures[[node_id]].to_numpy().flatten() for node_id in pressure_nodes]).T

        X_final, Y_final, _ = global_preprocessing(X, y)

        np.savez(f"{os.path.join(dir_out, str(i))}.npz", X_final=X_final, Y_final=Y_final)




if __name__ == "__main__":
    prepare_clean_hanoi_data(path_to_data="LeakDB/Hanoi_CMH/")
