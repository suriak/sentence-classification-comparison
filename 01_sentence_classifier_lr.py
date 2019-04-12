from time import time
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.utils import shuffle
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
import numpy as np
import pandas as pd


training_data = "./data-set/trec10_train.csv"
test_data = "./data-set/trec10_test.csv"
df_train = shuffle(pd.read_csv(training_data, delimiter=','))
X_train, Y_train = df_train['Query'], df_train['Type']
df_test = shuffle(pd.read_csv(test_data, delimiter=','))
X_test, Y_test = df_test['Query'], df_test['Type']

propertyClassifier = Pipeline([
        ('vectorizer', CountVectorizer(stop_words=None)),
        ('model', SGDClassifier(loss='log', alpha=1e-2, max_iter=10000, n_jobs=-1)),
    ])
parameters = {
    'vectorizer__ngram_range': [(1, 1), (2, 2)],
    'vectorizer__binary': (True, False),
    'model__penalty': ('l2', 'l1'),
}

times = list()
train_accuracies = list()
test_accuracies = list()
epoch = 5
for i in range(epoch):
    print("*" * 25)
    print("Run %s/%s" % (i+1, epoch))
    print("Training set: %s" % training_data)
    print("Performing grid search...")
    train_start_time = time()
    gridSearchClassifier = GridSearchCV(propertyClassifier, parameters)
    gridSearchClassifier = gridSearchClassifier.fit(X_train, Y_train)
    train_end_time = time()
    time_taken = round((train_end_time - train_start_time), 2)
    print("Training time taken is %s units." % (str(time_taken)))
    times.append(time_taken)
    best_score = round(gridSearchClassifier.best_score_, 2)
    print("Best score: %s" % best_score)
    train_accuracies.append(best_score)
    # print "Best parameters \n%s"%gridSearchClassifier.best_params_

    print("Test set: %s" % test_data)
    prediction = gridSearchClassifier.predict(X_test)
    test_accuracy = round(np.mean(Y_test == prediction).item(), 2)
    print("Accuracy of model for test set is %s\n" % test_accuracy)
    test_accuracies.append(test_accuracy)

print("Training Times(in Sec.): ")
print(times)
print("Training Accuracies:")
print(train_accuracies)
print("Test Accuracies: ")
print(test_accuracies)
