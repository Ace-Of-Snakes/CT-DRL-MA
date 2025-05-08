
import numpy as np
import pickle

def load_kde_model(model_path):
    with open(model_path, 'rb') as f:
        return pickle.load(f)

def sample_from_kde(kde_model, n_samples=1, min_val=0, max_val=24):
    samples = kde_model.sample(n_samples=n_samples)
    
    # Apply modulo 24 to ensure time values stay within 0-24 range
    if min_val == 0 and max_val == 24:
        samples = samples % 24
    
    # Clip values to specified range
    samples = np.clip(samples, min_val, max_val)
    
    return samples.flatten()

def hours_to_time(hours):
    h = int(hours)
    m = int((hours - h) * 60)
    s = int(((hours - h) * 60 - m) * 60)
    return h, m, s
