# %%
import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from models import NSW

# %%

data = np.load("../input/plarge300.npy", allow_pickle=True).item()
X_train, X_test, y_train, y_test = train_test_split(data['X'], data['y'],
                                                    test_size=0.5, random_state=1992)

# %%

param_dict = {
    'f': np.arange(3, 11, 2),
    'm': np.arange(3, 21, 2),
    'k': np.arange(1, 10, 2)
}

# %%

# nsw = NSW()
# gscv = GridSearchCV(nsw, param_dict, cv=10, scoring="accuracy", n_jobs=-1)
# gscv.fit(X_train, y_train)
# best_param = gscv.best_params_
# best_score = gscv.best_score_
# %%
best_param = {'f': 3, 'k': 5, 'm': 15}
best_score = 0.7866666666666666
print("Best Parameters: ", best_param)
print("Best Accuracy: ", best_score)

# %%

nsw = NSW(**best_param, metric="lb_keogh", metric_params={"radius": 23})
nsw.fit(X_train, y_train)
y_hat = nsw.predict(X_test)
acc1 = accuracy_score(y_hat, y_test)
print("Model accuracy: ", round(acc1, 2))
acc2 = nsw.score(X_test, y_test)
print("Model accuracy: ", round(acc2, 2))
# assert acc1 == acc2
# # %%
