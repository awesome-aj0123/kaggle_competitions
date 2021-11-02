import sys
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix


def plot_df(df, subsets):
    """ preferably 2 dimensional subset vector for visualization, 1 for class
    [x, y, class]
    """
    x_data = df[subsets[0]]
    y_data = df[subsets[1]]
    c_data = df[subsets[2]]

    plt.scatter(x_data, y_data, c=c_data)
    plt.xlabel(f'{subsets[0]}')
    plt.ylabel(f'{subsets[1]}')
    plt.show()


def dropna(df, axis=0, inplace=True, subset=None):
    df.dropna(axis=axis, inplace=inplace, subset=subset)


def solve_titanic(input_file, output_file):
    df = pd.read_csv(f'../data/{input_file}.csv')

    subset = ["Survived", "Pclass", "Sex", "Age"]

    # tailor the data to remove NaN values and consider only subset solumns
    # CHANGES FOR ALL COLUMNS
    dropna(df, axis=0, inplace=True, subset=subset)
    df = df[subset]
    df.replace({"male": 0, "female": 1}, inplace=True)

    # plot_df(df, ['Pclass', 'Age', 'Survived'])

    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20)

    svc_classifier = SVC(kernel='rbf', degree=3)
    svc_classifier.fit(X_train, y_train)

    y_pred = svc_classifier.predict(X_test)

    print(confusion_matrix(y_test, y_pred))
    print(classification_report(y_test, y_pred))


if __name__ == '__main__':
    """ Titanic Kaggle task

    Author: Abhishek Joshi
    """

    input_file = sys.argv[1]
    output_file = sys.argv[2]

    solve_titanic(input_file=input_file,
                  output_file=output_file)
