from sklearn import tree
from binary_classification import load_data
from sklearn.metrics import accuracy_score, classification_report

'''
Chosen model: Decision Tree Classifier
A decision tree is a supervised learning algorithm with a tree structure consisting of a root node, 
decision nodes, and leaf nodes. The algorithm learns to classify by finding the feature thresholds that
 best separate the classes in the training data. At each internal node, the sample is routed left or right 
 depending on whether a feature value falls above or below a learned threshold. This process repeats down the tree 
 until the sample reaches a leaf node, which outputs the predicted class label.'''

X_train, X_test, y_train, y_test, feature_names = load_data()
print(f"Training samples: {len(y_train)}")
print(f"Test samples: {len(y_test)}")
print(f"Features: {X_train.shape[1]}")


clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred, target_names=['malignant', 'benign']))

'''
Decision tree accuracy: 0.9386
from-scratch model accuracy: 0.9912

The from-scratch model (logistic regression) performed better. I suspect that the from-scratch model performed better because
it is similar to a logistic regression model which assumes linear separability (or close to linear). Decision tree on the other hand,
has no linearity assumption thus has a "hierarchical if-else" like structure. On a binary classification task like this one,
I suspect that the task has a clearer linear seperability which the logistic model is better suited for. ( also maybe the 
the decision tree is overfitting the training data? online it says to use pruning to prevent overfitting but I haven't tried 
that so not sure if if it applies in this context)

#=============================================================


Decision tree OUTPUT: 
Training samples: 455
Test samples: 114
Features: 30
Accuracy: 0.9386
              precision    recall  f1-score   support

   malignant       0.91      0.93      0.92        43
      benign       0.96      0.94      0.95        71

    accuracy                           0.94       114
   macro avg       0.93      0.94      0.93       114
weighted avg       0.94      0.94      0.94       114


from-scratch model OUTPUT: 
Training samples: 455
Test samples: 114
Features: 30

Testing sigmoid...
  sigmoid(0) = 0.5000 (should be 0.5)
  sigmoid(10) = 1.0000 (should be ~1.0)

Training...
Epoch   0, Loss: 0.0545
...
Epoch  90, Loss: 0.0080

Training accuracy: 0.9868
Test accuracy: 0.9912

'''