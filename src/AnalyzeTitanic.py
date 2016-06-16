import pandas

titanic = pandas.read_csv("train.csv")

titanic["Age"] = titanic["Age"].fillna(titanic["Age"].median())
titanic["Embarked"] = titanic["Embarked"].fillna("S")
titanic.loc[titanic["Embarked"] == "S", "Embarked"] = 0
titanic.loc[titanic["Embarked"] == "C", "Embarked"] = 1
titanic.loc[titanic["Embarked"] == "Q", "Embarked"] = 2
# print titanic.head()

print titanic["Survived"].unique()
print titanic["Pclass"].unique()
print titanic["Age"].unique()
print titanic["SibSp"].unique()
print titanic["Parch"].unique()
print titanic["Fare"].unique()