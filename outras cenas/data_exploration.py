import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree

hhd = pd.read_csv("hungarian_heart_diseases.csv")
X, y = hhd.drop('outcome', axis=1), hhd['outcome']


X_train, X_temp, y_train, y_temp = train_test_split(X, y, train_size=0.6, stratify=y, random_state=1)
X_test, X_validation, y_test, y_validation = train_test_split(X_temp, y_temp, train_size=0.5, random_state=1)

max_depth = [2, 4]
sample_leaf_sizes = [2, 100]
values = []

best_model = None

for depth in range(2,4):
    for leaf in range(2,100):
        classifier = DecisionTreeClassifier(min_samples_leaf=leaf, max_depth=depth, random_state=1)
        classifier.fit(X_train, y_train)

        t_acc = classifier.score(X_test, y_test)
        v_acc = classifier.score(X_validation, y_validation)
        
        if t_acc >= 0.785 and v_acc >= 0.8:
            if best_model == None or (t_acc+v_acc) > (best_model[0]+best_model[1]):
                best_model = (depth, leaf)
    
if best_model == None:
    print("No model fits parameters.")
else:
    classifier = DecisionTreeClassifier(min_samples_leaf=best_model[1], max_depth=best_model[0], random_state=1)
    plot_tree(
    classifier,
    feature_names=X.columns,
    class_names=["Normal", "Heart Disease"],
    filled=True,
    rounded=True,
    fontsize=10,
    )
    plt.title(f"Best Decision Tree (depth={best_model[0]}, min_samples_split={best_model[1]})")
    plt.show()
