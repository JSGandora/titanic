import pandas
# Import the linear regression class
from sklearn.linear_model import LogisticRegression
# Sklearn also has a helper that makes it easy to do cross validation
from sklearn.cross_validation import KFold
import numpy as np

titanic = pandas.read_csv("train.csv")

# Preprocess data
titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
titanic.loc[titanic["Sex"] == "male", "Sex"] = 1
titanic.loc[titanic["Sex"] == "female", "Sex"] = 0

# Manipulate features
titanic["Embarked_squared"] = titanic["Embarked"]**2
titanic["Age_squared"] = titanic["Age"]**2
print titanic.head()

# The columns we'll use to predict the target
predictors = ["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare", "Embarked", "Embarked_squared"]

# Initialize our algorithm class
alg = LogisticRegression()
# Generate cross validation folds for the titanic dataset.  It return the row indices corresponding to train and test.
# We set random_state to ensure we get the same splits every time we run this.
kf = KFold(titanic.shape[0], n_folds=3, random_state=1)

predictions = []
for train, test in kf:
    # The predictors we're using the train the algorithm.  Note how we only take the rows in the train folds.
    train_predictors = titanic[predictors].iloc[train,:]
    # The target we're using to train the algorithm.
    train_target = titanic["Survived"].iloc[train]
    # Training the algorithm using the predictors and target.
    alg.fit(train_predictors, train_target)
    # We can now make predictions on the test fold
    test_predictions = alg.predict(titanic[predictors].iloc[test,:])
    predictions.append(test_predictions)

# The predictions are in three separate numpy arrays.  Concatenate them into one.
# We concatenate them on axis 0, as they only have one axis.
predictions = np.concatenate(predictions, axis=0)

# accuracy = np.sum(predictions[predictions == titanic["Survived"]]) / float(len(predictions))
# print accuracy

count = 0
for idx in range(0, len(predictions)):
    if predictions[idx] == titanic["Survived"][idx]:
        count += 1
accuracy = count / float(len(predictions))
print accuracy