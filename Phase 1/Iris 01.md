Iris 1 Notes (Written by Anthony)
 # import pandas as pd
 #Pandas is a fast, powerful, flexible and easy to use open source data analysis and manipulation tool, built on top of the Python programming language.
# import numpy as np 
 #Numpy is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
# import matplotlib.pyplot as plt
    #Matplotlib is a comprehensive library for creating static, animated, and interactive visualizations in Python.
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
 - Sklearn is a Open Source Libary for Machine Learning, Model Selection gives us access to various models
 - Train_Test_Split = Split arrays or matrices into random train and test subsets.
 - StratifiedKFold = Class-wise stratified K-Fold cross-validator.
  Provides train/test indices to split data in train/test sets. This cross-validation object is a variation of KFold that returns stratified folds. The folds are made by preserving the percentage of samples for each class in y in a binary or multiclass classification setting.
- Cross_val_score = Evaluate a score by cross-validation. 

## Step 1 
- We pulled in the iris dataset and put it into a table (DataFrame) so it's easy to work with. Before doing anything else, you checked its shape (150 rows, 4 columns of measurements), what those columns actually represent (sepal length, sepal width, petal length, petal width), and whether the three species are represented equally. This matters because if one species had way fewer samples than the others, your model could end up biased toward the bigger group. Iris turned out to be perfectly balanced: 50 samples each.

    # Sklearn datasets brings in popular datasets for machine learning, We imported the Iris Data Set
    from sklearn.datasets import load_iris
    # Import pandas gives us data manipulation and analysis tools
    import pandas as pd
    # Load the iris dataset
    iris = load_iris()
    # Create a DataFrame from the iris data
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    # Add the target column to the DataFrame, mapping numeric values to their corresponding species names
    df['species'] = iris.target_names[iris.target]
    # Display the DataFrame
    print("Shape: ", df.shape)
    # Display the feature names
    print("Feature names: ", iris.feature_names)
    # Display the first few rows of the DataFrame
    print("Class balance: ")
    # Display the class distribution
    print(df['species'].value_counts())
    # Display the first few rows of the DataFrame
    df.head()

## Step 2
- Numbers alone don't tell you much about how separable the classes are, so you plotted every feature against every other feature, colored by species. This is what the pairplot showed you: setosa is easy to tell apart from the other two on almost any feature, while versicolor and virginica overlap more. You found that petal length vs. petal width gave the cleanest separation between all three species. This visual check is useful because it gives you intuition for what a model should be able to learn before you even train one.

    # Seaborn is a data visualization library based on matplotlib that provides a high-level interface for drawing attractive and informative statistical graphics
    import seaborn as sns
    # Matplotlib is a plotting library for the Python programming language and its numerical mathematics extension NumPy. It provides an object-oriented API for embedding plots into applications using general-purpose GUI toolkits like Tkinter, wxPython, Qt, or GTK.
    import matplotlib.pyplot as plt
    # Create a pairplot of the DataFrame, coloring by species and using histograms for the
    sns.pairplot(df, hue='species', diag_kind='hist')
    # Show the plot
    plt.show()
    
    (Image shown in the 01_iris)

## Step 3
- Before training anything, you need to hold back some data to test on later, data the model never sees during training. That's what train_test_split does. The word "stratified" means the split keeps the same class proportions in both the training set and the test set. Without it, random chance could leave you with a test set that's missing a species entirely, or badly imbalanced, which would give you a misleading picture of performance.

    # Model_Selection is a process of selecting the best model from a set of candidate models based on their performance on a given dataset. It involves evaluating the models using various metrics and selecting the one that performs the best.
    from sklearn.model_selection import train_test_split

    # Features
    X = df[iris.feature_names]
    # Target variable
    Y = df['species']

    # Split the dataset into training and testing sets, with 20% of the data reserved for testing
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)

    # Display the shape of the training and testing sets
    print("Train Class balance: \n", Y_train.value_counts())
    # Display the class distribution in the test set
    print("Test Class balance: \n", Y_test.value_counts())

## Step 4
- A baseline is the simplest reasonable model you can fit, here, logistic regression. The point isn't that logistic regression is necessarily the best choice, it's to establish a floor. Any fancier model or technique you try afterward should be compared against this baseline to see if it's actually adding value. Your baseline scored perfectly (100% accuracy) on the test set, which tells you this dataset is unusually easy to classify. That sets expectations for the later steps: with such a strong baseline, cross-validation and comparing model families is less about chasing higher accuracy and more about checking how consistent and reliable that performance is.

    # Linear_models is a module in scikit-learn that provides various linear models for regression and classification tasks.
    # LogisticRegression is a linear model for classification that estimates the probability of a binary response based on one or more predictor variables.
    from sklearn.linear_model import LogisticRegression
    # Sklearn metrics module provides functions to evaluate the performance of machine learning models.
    from sklearn.metrics import accuracy_score, classification_report

    # Create a baseline logistic regression model with a maximum of 1000 iterations
    baseline = LogisticRegression(max_iter=1000)
    # Fit the baseline model to the training data
    baseline.fit(X_train, Y_train)

    # Make predictions on the test set using the baseline model
    y_pred = baseline.predict(X_test)

    # Calculate and print the accuracy score of the baseline model on the test set
    print("Accuracy: ", accuracy_score(Y_test, y_pred))
    # Calculate and print the classification report, which includes precision, recall, f1-score, and support for each class
    print("\n", classification_report(Y_test, y_pred))

## Step 5
- A single train/test split only tells you how the model does on one particular slice of data. Maybe that slice happened to be easy, or maybe it happened to be hard. You don't really know. Cross-validation fixes this by testing the model multiple times on different slices. StratifiedKFold splits your data into 5 chunks (folds), each keeping the same class proportions as the full dataset. Then the process rotates: train on 4 folds, test on the 1 left out, repeat 5 times so every fold gets a turn as the test set. You end up with 5 accuracy scores instead of 1. The mean tells you the model's typical performance. The standard deviation tells you how much that performance swings depending on which data it sees, a low std means the model is stable and reliable, a high std means it's sensitive to which rows end up in training vs. testing.

    # StratifiedKFold is a cross-validator that provides train/test indices to split data into train/test sets while preserving the percentage of samples for each class.
    # Cross_val_score is a function that evaluates a score by cross-validation, splitting the dataset into k folds and returning the scores for each fold.
    from sklearn.model_selection import StratifiedKFold, cross_val_score

    # Create a StratifiedKFold cross-validator with 5 splits, shuffling the data and setting a random state for reproducibility
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    # Perform cross-validation using the baseline model, features (X), target variable (Y), and the StratifiedKFold cross-validator
    cv_scores = cross_val_score(baseline, X, Y, cv=skf)

    # Print the cross-validation scores, mean score, and standard deviation of the scores
    print("CV Scores: ", cv_scores)
    # Print the mean of the cross-validation scores
    print("Mean CV Score: ", cv_scores.mean())
    # Print the standard deviation of the cross-validation scores
    print("Standard Deviation of CV Scores: ", cv_scores.std())

## Step 6
- Our baseline was logistic regression, which draws relatively simple, smooth boundaries between classes. But different algorithms "think" differently: KNN classifies a point by looking at its nearest neighbors and taking a majority vote. It makes no assumptions about the shape of the boundary between classes. Decision Tree splits the data by asking a series of yes/no questions on individual features (like "is petal length > 2.5?"), building boundaries that look like rectangles rather than smooth curves. Running all three through the same cross-validation setup lets you compare them on equal footing. If one model consistently outperforms the others, or if one is far more stable (lower std) across folds, that tells you something about which approach fits the structure of this data best. Since iris is such a clean, separable dataset, you may find all three land in a similar range. That in itself is informative: it tells you the choice of algorithm matters less here than the fact that your features are just genuinely good at separating the classes.

    # Sklearn Neighbors module provides functionality for implementing the k-nearest neighbors algorithm, which is a simple,  instance-based learning method used for classification and regression tasks.
    # KNeighborsClassifier is a class that implements the k-nearest neighbors algorithm for classification tasks, allowing you to fit a model to your data and make predictions based on the majority class of the nearest neighbors.
    from sklearn.neighbors import KNeighborsClassifier
    # DecisionTreeClassifier is a class that implements the decision tree algorithm for classification tasks, allowing you to fit a model to your data and make predictions based on the learned decision rules
    from sklearn.tree import DecisionTreeClassifier

    models = {
        "LogisticRegression": LogisticRegression(max_iter=1000),
        "KNeighborsClassifier": KNeighborsClassifier(),
        "DecisionTreeClassifier": DecisionTreeClassifier()
    }

    # Perform cross-validation for each model and print the mean and standard deviation of the scores
    for name, model in models.items():
        #Create a StratifiedKFold cross-validator with 5 splits, shuffling the data and setting a random state for reproducibility
        scores = cross_val_score(model, X, Y, cv=skf)
        #Print the mean and standard deviation of the cross-validation scores for each model
        print(f"{name}: mean={scores.mean():.4f}, std={scores.std():.4f}")

## Step 7
- Accuracy gives you one number for the whole model, but it can hide problems. Imagine a model that's great at spotting setosa but keeps mixing up versicolor and virginica, accuracy alone wouldn't tell you that, it would just average everything into one score. The confusion matrix breaks predictions down by class: rows are the true labels, columns are what the model predicted. Your matrix (10, 9, 11 down the diagonal, zeros everywhere else) means every single prediction landed on the correct diagonal cell, no confusion at all between species. The classification report goes further, giving you precision, recall, and f1-score per class. Precision asks "of everything the model called virginica, how much actually was virginica?" Recall asks "of everything that actually was virginica, how much did the model catch?" Looking at both matters because a model can have high overall accuracy while still being weak on a particular class, especially in problems with imbalanced classes. Here, since you picked KNN as your best model, this step confirmed it's not just accurate overall, it's accurate everywhere.
    # Metrics module in scikit-learn provides functions to evaluate the performance of machine learning models.
    # Confusion matrix is a table that is often used to describe the performance of a classification model on a set of test data for which the true values are known.
    # Classification report is a text summary of the precision, recall, f1-score, and support for each class in a classification model.
    from sklearn.metrics import confusion_matrix, classification_report

    # KNeighborsClassifier is a class that implements the k-nearest neighbors algorithm for classification tasks
    best_model = KNeighborsClassifier()
    # Fit the best model to the training data
    best_model.fit(X_train, Y_train) 
    # Make predictions on the test set using the best model
    y_pred = best_model.predict(X_test) 

    # Print the confusion matrix for the test set predictions
    print(confusion_matrix(Y_test, y_pred)) 
    # Print the classification report for the test set predictions
    print(classification_report(Y_test, y_pred)) 

## Step 8
- Up to this point, your test set had been touched more than once. You checked baseline accuracy on it, then compared it against KNN and decision tree during model selection. Every time you look at test performance and use that information to make a decision (like which model to choose), you leak a little bit of information from the test set into your decision-making process. Technically that means your test set stops being a perfectly clean, unbiased judge. This step is your one final, official check: run the chosen model against the test set one last time and record that number as the true, honest estimate of how it'll perform on new data. The rule of thumb is you touch the test set once for this final judgment and then stop, you don't go back and tweak the model based on this result, because that would start the leakage problem all over again. In your case it came out to 100% again, which makes sense given how separable the classes are, but the discipline of this step matters more on harder, real-world datasets where the final number might differ from what you saw during model comparison.
    # Make predictions on the entire dataset using the best model
    y_pred_final = best_model.predict(X_test) 
    # Print the final predictions for the entire dataset
    print("Final Predictions: ", y_pred_final) 
    # Print the confusion matrix for the entire dataset predictions
    print("\n", confusion_matrix(Y_test, y_pred_final)) 
    # Print the classification report for the entire dataset predictions
    print("\n", classification_report(Y_test, y_pred_final)) 
