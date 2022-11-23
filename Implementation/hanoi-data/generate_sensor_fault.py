import os
import numpy as np
import random


def zero(X, faulty_time, faulty_idx):
    for t in faulty_time:
        X[t, faulty_idx] = 0.0
    return X

def noisy(X, faulty_time, faulty_idx, sigma=1.):
    for t in faulty_time:
        X[t, faulty_idx] += np.random.randn() * sigma
    return X

def offset(X, faulty_time, faulty_idx, offset=5.):
    for t in faulty_time:
        X[t, faulty_idx] += offset
    return X

def shift(X, faulty_time, faulty_idx, a=.5, b=5):
    for t in faulty_time:
        X[t, faulty_idx] = X[t, faulty_idx] * a + b
    return X


def random_failure_type():
    return random.choice(["noisy", "offset", "shift"])


def apply_sensor_failure(X, failure_type, faulty_time, faulty_idx, params):
    if failure_type == "noisy":
        return noisy(X, faulty_time, faulty_idx, **params)
    elif failure_type == "offset":
        return offset(X, faulty_time, faulty_idx, **params)
    elif failure_type == "shift":
        return shift(X, faulty_time, faulty_idx, **params)
    else:
        raise ValueError(f"Unknown failure type '{failure_type}'")


def random_params(failure_type):
    if failure_type == "noisy":
        return {"sigma": random.choice(range(1, 10, 1))}
    elif failure_type == "offset":
        return {"offset": random.choice(range(5, 10, 1))}
    elif failure_type == "shift":
        return {"a": random.random(), "b": random.choice(range(5, 10, 1))}
    else:
        raise ValueError(f"Unknown failure type '{failure_type}'")


def random_fault_time(X):
    time_win_len = random.randint(100, 1000)
    start_time = random.randint(int(X.shape[0] / 2), X.shape[0] - 2 * time_win_len)

    return [start_time + t for t in range(time_win_len)]


def select_random_sensor(X):
    return random.choice(range(X.shape[1]))


def generate_random_failure(X):
    failure_type = random_failure_type()
    faulty_time = random_fault_time(X)
    faulty_sensor_idx = select_random_sensor(X)
    fault_params = random_params(failure_type)

    X = apply_sensor_failure(X, failure_type, faulty_time, faulty_sensor_idx, fault_params)
    
    y_faulty = np.zeros(X.shape[0])
    for t in faulty_time:
        y_faulty[t] = 1
    
    return X, y_faulty


if __name__ == "__main__":
    path_in = "hanoi_clean/"
    dir_out = "hanoi_faultysensor/"

    files_in = [f_in for f_in in os.listdir(path_in) if f_in.endswith(".npz")]

    for f_in, i in zip(files_in, range(len(files_in))):
        print(f_in)
        data = np.load(os.path.join(path_in, f_in))
        X_final, Y_final = data["X_final"], data["Y_final"]

        X_final, y_faulty = generate_random_failure(X_final)
        np.savez(f"{os.path.join(dir_out, str(i))}.npz", X_final=X_final, Y_final=Y_final, y_faulty=y_faulty)
