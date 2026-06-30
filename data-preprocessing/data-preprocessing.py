import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def main():
    # Load the dataset
    dataset = pd.read_csv("Data.csv")

    # Split the dataset into features (X) and target (y)
    X = dataset.iloc[:, :-1].values
    y = dataset.iloc[:, -1].values

    print("Original Features:\n", X)
    print("\nOriginal Target:\n", y)

    # Replace missing values in numerical columns with the mean
    imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
    X[:, 1:3] = imputer.fit_transform(X[:, 1:3])

    print("\nFeatures after handling missing values:\n", X)

    # One-hot encode the Country column
    ct = ColumnTransformer(
        transformers=[("encoder", OneHotEncoder(), [0])],
        remainder="passthrough"
    )

    X = ct.fit_transform(X)
    print("\nFeatures after one-hot encoding:\n", X)

    # Encode the target variable
    le = LabelEncoder()
    y = le.fit_transform(y)

    print("\nEncoded Target:\n", y)

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=1
    )

    print("\nTraining Features:\n", X_train)
    print("\nTesting Features:\n", X_test)
    print("\nTraining Labels:\n", y_train)
    print("\nTesting Labels:\n", y_test)

    # Scale only the numerical features (Age and Salary)
    sc = StandardScaler()
    X_train[:, 3:] = sc.fit_transform(X_train[:, 3:])
    X_test[:, 3:] = sc.transform(X_test[:, 3:])

    print("\nScaled Training Features:\n", X_train)
    print("\nScaled Testing Features:\n", X_test)

    print("\nData preprocessing completed successfully!")

if __name__ == "__main__":
    main()