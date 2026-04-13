# aklearn.py

def normalize(data):
    """Scale values between 0 and 1."""
    min_val = min(data)
    max_val = max(data)
    return [(x - min_val) / (max_val - min_val) for x in data]

def accuracy(y_true, y_pred):
    """Calculate accuracy percentage."""
    correct = sum(t == p for t, p in zip(y_true, y_pred))
    return correct / len(y_true) * 100