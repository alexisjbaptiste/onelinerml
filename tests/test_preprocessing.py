import numpy as np
import pandas as pd
from onelinerml.preprocessing import preprocess_data


def test_preprocess_no_data_leakage():
    train_df = pd.DataFrame({
        'num': [1, 2, 3],
        'cat': ['A', 'B', 'B'],
        'target': [0, 1, 0]
    })

    test_df = pd.DataFrame({
        'num': [4],
        'cat': ['C'],
        'target': [1]
    })

    X_train, y_train, X_test, y_test, preprocessor = preprocess_data(
        train_df, 'target', test_df
    )

    cats = preprocessor.named_transformers_['cat']['onehot'].categories_[0].tolist()
    assert cats == ['A', 'B']

    # Category 'C' should be unseen and encoded as all zeros
    cat_features = X_test[:, -len(cats):]
    assert np.all(cat_features == 0)
