import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split, GridSearchCV

from models import HNSWClassifier

data = np.load("../input/plarge300.npy", allow_pickle=True).item()
X_train, X_test, y_train, y_test = train_test_split(data["X"], data["y"], test_size=0.5, random_state=1992)

param_grid = {
    "n_neighbors": np.arange(1, 11, 2),
    "max_elements": np.arange(10, 50, 10),
    "ef_Search": np.arange(10, 50, 10),
    "ef_construction": np.arange(100, 200, 20)
}
hnsw = HNSWClassifier(space="l2",
                      M=5,
                      random_seed=1992,
                      num_threads=4)

gscv = GridSearchCV(hnsw, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
gscv = gscv.fit(X_train, y_train)

best_param = gscv.best_params_
best_score = gscv.best_score_

print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_score)

hnsw = HNSWClassifier(**best_param,
                      space="l2",
                      M=5,
                      random_seed=1992,
                      num_threads=4).fit(X_train, y_train)
y_hat = hnsw.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Model accuracy w/o Mac-Fac: ", round(acc, 2))

param_grid = {
    "n_neighbors": np.arange(1, 11, 2),
    "mac_neighbors": np.arange(20, 50, 10),
    "max_elements": np.arange(10, 50, 10),
    "ef_Search": np.arange(10, 50, 10),
    "ef_construction": np.arange(100, 200, 20)
}

hnsw = HNSWClassifier(space="l2",
                      M=5,
                      metric_params={"global_constraint": "sakoe_chiba",
                                     "sakoe_chiba_radius": 23},
                      random_seed=1992,
                      num_threads=4)

gscv = GridSearchCV(hnsw, param_grid, cv=10, scoring="accuracy", n_jobs=-1)
gscv = gscv.fit(X_train, y_train)

best_param = gscv.best_params_
best_score = gscv.best_score_

print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_score)

hnsw = HNSWClassifier(**best_param,
                      space="l2",
                      M=5,
                      metric_params={"global_constraint": "sakoe_chiba",
                                     "sakoe_chiba_radius": 23},
                      random_seed=1992,
                      num_threads=4).fit(X_train, y_train)
y_hat = hnsw.predict(X_test)
acc = accuracy_score(y_test, y_hat)
print("Model accuracy w/ Mac-Fac: ", round(acc, 2))
