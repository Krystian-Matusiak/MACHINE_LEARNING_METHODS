import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
# from sklearn.model_selection import learning_curve, ShuffleSplit, LearningCurveDisplay
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
import seaborn as sns
from sklearn.svm import SVC



if __name__ == '__main__':


    columns = ['Sepal Length', 'Sepal Width', 'Petal Length', 'Petal Width', 'Class Labels']
    df = pd.read_csv('iris.csv', names=columns, header=None)
    print(f"Iris dataframe: \n {df}")

    # Visualize the whole dataset
    # b = sns.pairplot(df, hue='Class Labels', palette='plasma', markers=["o", "*", "h"])
    # plt.subplots()
    # plt.show()

    # d = sns.pairplot(df, hue='Class Labels', palette='plasma', kind="kde")
    # plt.show()

    # Seperate features and target
    data = df.values
    # X data equal to values
    X = data[:, 0:4]
    # Y data equal to flower types
    Y = data[:, 4]

    # Calculate avarage of each features for all classes
    Y_Data = np.array([np.average(X[:, i][Y == j].astype('float32')) for i in range(X.shape[1]) for j in (np.unique(Y))])
    Y_Data_reshaped = Y_Data.reshape(4, 3)
    Y_Data_reshaped = np.swapaxes(Y_Data_reshaped, 0, 1)
    X_axis = np.arange(len(columns) - 1)
    width = 0.25
    print("Y Data")
    print(Y_Data_reshaped)

    # Plot the avarage
    plt.bar(X_axis, Y_Data_reshaped[0], width, label='Setosa', color='darkorchid')
    plt.bar(X_axis + width, Y_Data_reshaped[1], width, label='Versicolour', color='hotpink')
    plt.bar(X_axis + width * 2, Y_Data_reshaped[2], width, label='Virginica', color='lightsalmon')
    plt.xticks(X_axis, columns[:4])

    plt.xlabel("Leafes Features")
    plt.ylabel("Value in cm.")
    plt.legend(bbox_to_anchor=(1.3, 1))
    # plt.subplots(2)
    plt.show()

    # Split the data to train and test dataset.
    # Compare two algorithms
    # from sklearn.model_selection import train_test_split, learning_curve



    naive_bayes = GaussianNB()
    svc = SVC(kernel="rbf", gamma=0.001)

    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6), sharey=True)

    common_params = dict(X=X, y=Y, train_sizes=np.linspace(0.1, 1.0, 5),
                        cv=ShuffleSplit(n_splits=100, test_size=0.2, random_state=0), score_type="both", n_jobs=10,
                        line_kw={"marker": "*"}, std_display_style="fill_between", score_name="Accuracy", )

    for ax_idx, estimator in enumerate([naive_bayes, svc]):
        LearningCurveDisplay.from_estimator(estimator, **common_params, ax=ax[ax_idx])
        handles, label = ax[ax_idx].get_legend_handles_labels()
        ax[ax_idx].legend(handles[:2], ["Training Score", "Test Score"])
        ax[ax_idx].set_title(f"Learning Curve for {estimator.__class__.__name__}")

    common_params = {
        "X": X,
        "y": Y,
        "train_sizes": np.linspace(0.1, 1.0, 5),
        "cv": ShuffleSplit(n_splits=100, test_size=0.2, random_state=0),
        "n_jobs": 10,
        "return_times": True,
    }

    train_sizes, train_scores_nb, test_scores_nb, fit_times_nb, score_times_nb = learning_curve(
        naive_bayes, **common_params
    )
    train_sizes, train_scores_svm, test_scores_svm, fit_times_svm, score_times_svm = learning_curve(
        svc, **common_params
    )

    print(f"{train_sizes} samples were used to train a model using GaussianNB")

    print(f"The average train accuracy for GaussianNB is {train_scores_nb.mean():.2f}")
    print(f"The average test accuracy for GaussianNB is {test_scores_nb.mean():.2f}")

    print(f"{train_sizes} samples were used to train a model using SVM")

    print(f"The average train accuracy for SVM is {train_scores_svm.mean():.2f}")
    print(f"The average test accuracy for SVM is {test_scores_svm.mean():.2f}")

    plt.show()

    print("Due to the better accuracy Gaussian method was used. ")

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12), sharex=True)

    for ax_idx, (fit_times, score_times, estimator) in enumerate(
        zip(
            [fit_times_nb, fit_times_svm],
            [score_times_nb, score_times_svm],
            [naive_bayes, svc],
        )
    ):

        # scalability regarding the fit time
        ax[0, ax_idx].plot(train_sizes, fit_times.mean(axis=1), "*-")
        ax[0, ax_idx].fill_between(
            train_sizes,
            fit_times.mean(axis=1) - fit_times.std(axis=1),
            fit_times.mean(axis=1) + fit_times.std(axis=1),
            alpha=0.3, color= 'purple')

        ax[0, ax_idx].set_ylabel("Fit time (s)")
        ax[0, ax_idx].set_title(
            f"Scalability of the {estimator.__class__.__name__} classifier"
        )

    # scalability regarding the score time
        ax[1, ax_idx].plot(train_sizes, score_times.mean(axis=1), "*-")
        ax[1, ax_idx].fill_between(
        train_sizes,
        score_times.mean(axis=1) - score_times.std(axis=1),
        score_times.mean(axis=1) + score_times.std(axis=1),
        alpha=0.3, color='pink'
    )
        ax[1, ax_idx].set_ylabel("Score time (s)")
        ax[1, ax_idx].set_xlabel("Number of training samples")

    plt.show()






    ###################################################################################################


    print("Splitting to individual points with GaussianNB")

    X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

    # Support vector machine algorithm


    clf = GaussianNB()
    clf.fit(X_train, y_train)
    print("Feeding Data into the algorithm")

    # svn = SVC()
    # svn.fit(X_train, y_train)
    # print("Feeding Data into the algorithm")

    print("Predictions")

    # Predictions
    print(clf.predict(X_test))

    predictions = clf.predict(X_test)
    print("Prediction of Species: {}".format(predictions))

    # Predict from the test dataset
    # predictions = svn.predict(X_test)
    # print("Predictions")


    # Calculate the accuracy


    accuracy_score(y_test, predictions)

    print(f"Accuracy Score, {accuracy_score(y_test, predictions)}")

    # detailed report - classification



    print(f"Classfification Report{classification_report(y_test, predictions)}")

    # Calculate the MSE
    XM = data[:, 0:3]
    YM = data[:, 3]
    YM = YM.astype('int')

    XM_train, XM_test, yM_train, yM_test = train_test_split(XM, YM, test_size=0.2)

    clfM = GaussianNB()
    clfM.fit(XM_train, yM_train)
    predictionsM = clfM.predict(XM_test)
    print(f"Predictions {clfM.predict(XM_test)}")

    # svnM = SVC()
    # svnM.fit(XM_train, yM_train)
    # predictionsM = svnM.predict(XM_test)



    mean_squared_error(yM_test, predictionsM)

    print(f"Mean squared error is {mean_squared_error(yM_test, predictionsM)}")

    # A detailed classification report
    # from sklearn.metrics import classification_report

    # print(classification_report(y_test, predictions))

    print("Random values to test the model")

    X_new = np.array([[3, 2, 1, 0.2], [4.9, 2.2, 3.8, 1.1], [5.3, 2.5, 4.6, 1.9]])
    # Prediction of the species from the input vector
    prediction = clf.predict(X_new)
    print("Prediction of Species: {}".format(prediction))

    #Plot realtion



    # Try Regression


    model = LogisticRegression(random_state=0)
    model.fit(X_train, y_train)

    Y_pred = model.predict(X_test)

    print(Y_pred)



    cm = confusion_matrix(y_test, Y_pred)

    print(confusion_matrix(y_test, Y_pred))

    print(accuracy_score(y_test, Y_pred))


    # confusion matrix sns heatmap
    ax = plt.axes()
    df_cm = cm
    sns.heatmap(df_cm, annot=True, annot_kws={"size": 30}, fmt='d', cmap="Reds", ax=ax)
    ax.set_title('Confusion Matrix')


    plt.show()

    from sklearn.metrics import confusion_matrix
    from mlxtend.plotting import plot_confusion_matrix

    matrix = confusion_matrix(y_test,Y_pred)

    class_names = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']

    fig, ax = plot_confusion_matrix(conf_mat=matrix,
                                    show_absolute=True,
                                    show_normed=False,
                                    colorbar=True,
                                    class_names=class_names)

    ax.set_title('Confusion Matrix')
    plt.show()

    #Try pytorch Neural Network

    import torch



