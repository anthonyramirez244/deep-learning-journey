Titanic 2 notes (Written by Anthony)

# import pandas as pd
# import numpy as np
# import matplotlib.pyplot as plt
# from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score

## Step 1: Load & audit
You pulled in the Titanic dataset via seaborn and loaded it into a DataFrame. The trickiest part here wasn't the data itself — it was your machine's SSL setup blocking the download (a certificate verification error), which you worked around by disabling the check. Once that was sorted, the dataset loaded cleanly: 891 passengers, 15 columns covering things like class, sex, age, fare, and whether they survived. This step is just "get the data in front of you and make sure it's actually usable" before doing any analysis.
-    # Ssl is a module that provides access to Transport Layer Security (previously and widely known as Secure Sockets Layer) encryption a and peer authentication facilities for network sockets, both client-side and server-side.
    import ssl
    # This is to fix the SSL certificate verification issue when loading datasets from seaborn
    import certifi
    # This line of code sets the default SSL context to an unverified context, which means that SSL certificate verification will be disabled. This is often done to bypass SSL certificate verification errors when making HTTPS requests, especially in environments where the SSL certificates may not be properly configured or trusted.
    ssl._create_default_https_context = ssl._create_unverified_context

    import seaborn as sns
    # Load the Titanic dataset from seaborn
    df = sns.load_dataset('titanic')
    # Display the first few rows of the dataset to understand its structure and contents
    df.head()

## Step 2: EDA by group
Before engineering anything or training a model, you checked whether the raw columns already hint at who survived. Grouping by sex showed a huge gap — 74.2% of women survived vs. 18.9% of men. Grouping by class showed a clear gradient — 63% survived in 1st class, 47.3% in 2nd, only 24.2% in 3rd. Age bands showed children did best (58%) and seniors did worst (22.7%). This step matters because it tells you, before you train anything, which signals a model should pick up on — if your later model completely ignores sex or class, that's a red flag.
- # Survival rate by sex
    print("Survival rate by sex:")
    # Group the dataset by 'sex' and calculate the mean survival rate for each group, rounding the results to three decimal places for better readability
    print(df.groupby('sex')['survived'].mean().round(3))

    # Survival rate by passenger class
    print("\nSurvival rate by class:")
    # Group the dataset by 'pclass' (passenger class) and calculate the mean survival rate for each class, rounding the results to three decimal places for better readability
    print(df.groupby('pclass')['survived'].mean().round(3))

    # Survival rate by age band
    df['age_band'] = pd.cut(df['age'], bins=[0, 12, 18, 35, 60, 100],
        labels=['Child', 'Teen', 'Adult', 'Middle Age', 'Senior'])
    # Group the dataset by 'age_band' and calculate the mean survival rate for each age band, rounding the results to three decimal places for better readability
    print("\nSurvival rate by age band:")
    # Group the dataset by 'age_band' and calculate the mean survival rate for each age band, rounding the results to three decimal places for better readability
    print(df.groupby('age_band', observed=True)['survived'].mean().round(3))

## Step 3: Engineer features
You built two new columns from what was already there: family_size (siblings/spouses + parents/children + the passenger themselves) and is_alone (whether family_size was 1). One caveat worth calling out: the classic Titanic exercise also extracts a Title (Mr/Mrs/Miss) from the passenger's name, but seaborn's version of this dataset doesn't include names at all, so that feature simply wasn't available — I flagged that instead of quietly skipping it. The two features you did build turned out meaningful: solo travelers survived at 30.4%, versus 50.6% for people traveling with family — being completely alone was worse.
- # Survival rate by family size, which is calculated as the sum of siblings/spouses (sibsp) and parents/children (parch) plus one (the passenger themselves)
    df['family_size'] = df['sibsp'] + df['parch'] + 1

    # Create a new column 'is_alone' to indicate whether a passenger is traveling alone (1) or with family (0)
    df['is_alone'] = (df['family_size'] == 1).astype(int)

    # Display the first few rows of the relevant columns to verify the calculations
    print(df[['sibsp', 'parch', 'family_size', 'is_alone']].head())

    # Survival rate by family size
    print("\nSurvival rate by family size:")
    # Group the dataset by 'family_size' and calculate the mean survival rate for each family size, rounding the results to three decimal places for better readability
    print(df.groupby('family_size')['survived'].mean().round(3))

    # Survival rate by is_alone
    print("\nSurvival rate by is_alone:")
    # Group the dataset by 'is_alone' and calculate the mean survival rate for passengers traveling alone versus those traveling with family, rounding the results to three decimal places for better readability
    print(df.groupby('is_alone')['survived'].mean().round(3))

## Step 4: Preprocessing pipeline
Real-world data is messy — some ages and embarkation ports are missing, and categories like sex or class aren't numbers a model can use directly. You built a ColumnTransformer that fills in missing numeric values with the median, missing categories with the most common value, scales the numbers, and one-hot-encodes the categories. Critically, you also split the data into training (712 passengers) and test (179 passengers) sets before any of this fitting happens — so the imputation and scaling statistics are only ever learned from training data, never leaking information from the test set.
- # Compose a preprocessing pipeline for the Titanic dataset, which includes both numeric and categorical features. The pipeline will handle missing values, scale numeric features, and one-hot encode categorical features.
    # Import ColumnTransformer is a class in scikit-learn that allows you to apply different preprocessing steps to different subsets of features in your dataset. It is particularly useful when you have a mix of numeric and categorical features that require different types of preprocessing.
    from sklearn.compose import ColumnTransformer
    # Pipeline module is a class in scikit-learn that allows you to create a sequence of data transformation and modeling steps that can be treated as a single object. It helps streamline the process of applying multiple transformations and fitting a model, making it easier to manage and reproduce your machine learning workflow.
    # Import Pipeline from sklearn.pipeline module. The Pipeline class allows you to create a sequence of data transformation and modeling steps that can be treated as a single object. It helps streamline the process of applying multiple transformations and fitting a model, making it easier to manage and reproduce your machine learning workflow.
    from sklearn.pipeline import Pipeline
    # Imputer is a class in scikit-learn that provides strategies for handling missing values in your dataset. It allows you to fill in missing values using various techniques, such as replacing them with the mean, median, or most frequent value of the feature.
    # Import SimpleImputer is a specific implementation of the Imputer class that provides simple strategies for imputing missing values. It can be used to fill in missing values with a constant value, the mean, median, or most frequent value of the feature.
    from sklearn.impute import SimpleImputer
    # Preprocessing module in scikit-learn provides various tools for transforming and scaling data before feeding it into machine learning models. It includes classes for encoding categorical variables, scaling numeric features, and handling missing values.
    # OneHotEncoder is a class in scikit-learn that converts categorical variables into a binary matrix representation, where each category is represented by a separate column with a value of 1 or 0 indicating the presence or absence of that category.
    # StandardScaler is a class in scikit-learn that standardizes features by removing the mean and scaling to unit variance. It transforms the data to have a mean of 0 and a standard
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    # Define the numeric and categorical features to be used in the preprocessing pipeline
    numeric_features = ['age', 'fare', 'sibsp', 'parch', 'family_size']
    categorical_features = ['pclass', 'sex', 'embarked', 'is_alone']

    # Split the dataset into training and test sets, with 80% of the data used for training and 20% for testing. The split is stratified based on the target variable 'survived' to ensure that both sets have a similar distribution of survival outcomes. A random seed is set for reproducibility.
    X = df[numeric_features + categorical_features]
    # The target variable 'survived' is extracted from the dataset to be used for training and evaluation of the machine learning models.
    y = df['survived']

    # Split the dataset into training and test sets, with 80% of the data used for training and 20% for testing. The split is stratified based on the target variable 'survived' to ensure that both sets have a similar distribution of survival outcomes. A random seed is set for reproducibility.
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Numeric pipeline: fill missing values (mostly 'age' and 'fare') with the median, then scale to zero mean and unit variance
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    # Categorical pipeline: fill missing values (mostly 'embarked') with the most frequent value, then one-hot encode
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))
    ])
    # Compose a preprocessing pipeline that applies the numeric and categorical transformers to their respective feature sets. The ColumnTransformer allows you to specify which transformer to apply to which subset of features, enabling you to handle both numeric and categorical data in a single pipeline.
    preprocessor = ColumnTransformer(transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ])
    # Apply the preprocessing pipeline to the training and test sets, transforming the features according to the specified steps in the numeric and categorical pipelines. The transformed data is returned as NumPy arrays, which can be used for training and evaluating machine learning models.
    print("Training set:", X_train.shape)
    print("Test set:", X_test.shape)

## Step 5: Naive baselines
Same idea as your logistic regression baseline in the Iris example, but here you used two dumb rules instead of a real model: "everyone dies" (61.5% accuracy, since ~62% of passengers actually died) and "all women survive, all men die" (77.7% accuracy). That second one is a strong floor — any real model needs to clear 77.7% to be worth using at all, since a single-column rule already gets you most of the way there.
- # Metrics module in scikit-learn provides functions for evaluating the performance of machine learning models. It includes various metrics for classification, regression, and clustering tasks, allowing you to assess how well your models are performing.
    # Accuracy score is a metric that measures the proportion of correctly predicted instances out of the total instances in a classification problem. It is calculated as the number of correct predictions divided by the total number of predictions, and it provides a simple way to evaluate the overall performance of a classification model.
    from sklearn.metrics import accuracy_score
    # Baseline 1: predict all passengers die.
    baseline_all_die = pd.Series(0, index=y_test.index)
    # Accuracy score is calculated by comparing the predicted values (all zeros) with the actual values in the test set. The accuracy score is then printed to provide a reference point for evaluating the performance of more complex models.
    acc_all_die = accuracy_score(y_test, baseline_all_die)
    # This line of code calculates the accuracy of the baseline model that predicts all passengers die (the majority class) by comparing the predicted values (all zeros) with the actual values in the test set. The accuracy score is then printed to provide a reference point for evaluating the performance of more complex models.
    print(f"'Everyone dies' baseline accuracy: {acc_all_die:.3f}")

    # Baseline 2: predict all women survive, all men die.
    baseline_sex_rule = (X_test['sex'] == 'female').astype(int)
    # Accuracy score is calculated by comparing the predicted values (based on the sex rule) with the actual values in the test set. The accuracy score is then printed to provide a reference point for evaluating the performance of more complex models.
    acc_sex_rule = accuracy_score(y_test, baseline_sex_rule)
    # This line of code calculates the accuracy of the baseline model that predicts all women survive and all men die by comparing the predicted values with the actual values in the test set. The accuracy score is then printed to provide a reference point for evaluating the performance of more complex models.
    print(f"'Women survive, men die' baseline accuracy: {acc_sex_rule:.3f}")

    # Any real model needs to clear both of these bars to justify its complexity

## Step 6: Compare model families
You trained three model types — logistic regression, random forest, and gradient boosting — each wrapped in a pipeline with the preprocessing from Step 4, and scored them with 5-fold stratified cross-validation (same "keep class proportions consistent across folds" idea as your stratified split in the Iris example, just applied to CV folds instead of a single train/test split). Results: logistic regression 79.6%, random forest 79.4%, gradient boosting 82.3%. All three clear the 77.7% baseline, but not by a huge margin — unlike Iris, this dataset is genuinely hard to get perfect, so the differences between models are modest and the "std" (variability across folds) matters as much as the average.
- # Linear models are a class of machine learning algorithms that assume a linear relationship between the input features and the target variable. They are simple, interpretable, and often perform well on linearly separable data. Examples include Linear Regression for regression tasks and Logistic Regression for binary classification tasks.
from sklearn.linear_model import LogisticRegression
# Enemble models are a class of machine learning algorithms that combine multiple individual models (often called "weak learners") to create a stronger overall model. The idea is that by aggregating the predictions of several models, the ensemble can achieve better performance and generalization than any single model alone. Examples of ensemble methods include Random Forests, Gradient Boosting, and AdaBoost.
# Random Forest is an ensemble learning method that constructs multiple decision trees during training and outputs the mode of the classes (classification) or mean prediction (regression) of the individual trees. It helps improve accuracy and control overfitting.
# Gradient Boosting is an ensemble learning technique that builds a series of weak learners (typically decision trees) in a sequential manner, where each subsequent model attempts to correct the errors of the previous models
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

# 5-fold stratified CV keeps the survived/died ratio consistent across every fold, which matters here since survival is imbalanced (~38% survived)
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Evaluate each model using cross-validation and store the results in a dictionary. The cross_val_score function performs the cross-validation, fitting the model on the training folds and evaluating it on the validation fold. The accuracy scores for each fold are collected, and the mean and standard deviation of the scores are printed for each model.
cv_results = {}
for name, model in models.items():
# Each pipeline refits its own preprocessor on every training fold, so imputation/scaling/encoding statistics never see the validation fold
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
# Perform cross-validation on the pipeline using the training data (X_train, y_train) and the specified cross-validation strategy (cv). The scoring parameter is set to 'accuracy' to evaluate the models based on their accuracy scores. The resulting accuracy scores for each fold are stored in the cv_results dictionary under the corresponding model name.
    scores = cross_val_score(pipe, X_train, y_train, cv=cv, scoring='accuracy')
# Store the cross-validation scores for the current model in the cv_results dictionary, using the model name as the key. The mean and standard deviation of the accuracy scores across the folds are then printed to provide an overview of the model's performance and variability.
    cv_results[name] = scores
# Print the mean and standard deviation of the accuracy scores for the current model, formatted to three decimal places. This provides a summary of the model's performance across the cross-validation folds, allowing for easy comparison between different models.
    print(f"{name}: {scores.mean():.3f} +/- {scores.std():.3f}")

## Step 7: Evaluate beyond accuracy
Accuracy alone can hide a model that's great at spotting deaths but bad at spotting survivors, so you looked at precision, recall, and F1 specifically for the "survived" class. All three models had noticeably lower recall on survivors (~65-71%) than on deaths — meaning they're more likely to miss an actual survivor than to miss an actual death. That's useful to know: if this were a real triage tool, that asymmetry would matter a lot more than the overall accuracy number.
- # Metrics module in scikit-learn provides functions for evaluating the performance of machine learning models. It includes various metrics for classification, regression, and clustering tasks, allowing you to assess how well your models are performing.
# Classification report is a function in scikit-learn that generates a text summary of the precision, recall, F1-score, and support for each class in a classification problem. It provides a comprehensive overview of the model's performance across different classes, allowing you to assess how well the model is performing for each class and identify any potential issues or imbalances in the predictions.
# Precision, recall, and F1-score are important metrics for evaluating classification models, especially in imbalanced datasets. Precision measures the proportion of true positive predictions among all positive predictions, recall measures the proportion of true positive predictions among all actual positives, and F1-score is the harmonic mean of precision and recall, providing a single metric that balances both aspects.
from sklearn.metrics import classification_report, precision_recall_fscore_support

# Refit each model family on the full training set, then evaluate once on the held-out test set (accuracy alone can hide a model that's great at spotting deaths but bad at spotting survivors, or vice versa)
last_pred = None
# Pipelines are used to streamline the process of applying multiple transformations and fitting a model, making it easier to manage and reproduce the machine learning workflow. In this case, each model is refitted on the full training set using a pipeline that includes the preprocessing steps and the classifier. The predictions are then made on the held-out test set, and precision, recall, and F1-score are calculated for the positive class (survived) to evaluate the model's performance.
for name, model in models.items():
    pipe = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', model)
    ])
# Fit the pipeline on the full training set (X_train, y_train) and make predictions on the held-out test set (X_test). The predicted values are stored in the variable y_pred, and the last model's predictions are saved in last_pred for later evaluation.
    pipe.fit(X_train, y_train)
    y_pred = pipe.predict(X_test)
    last_pred = y_pred
# Calculate precision, recall, and F1-score for the positive class (survived) using the precision_recall_fscore_support function. The average parameter is set to 'binary' to compute metrics for the positive class only, and pos_label is set to 1 to indicate that the positive class corresponds to passengers who survived. The results are printed for each model.
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='binary', pos_label=1
    )
# Print the precision, recall, and F1-score for the positive class (survived) for the current model, formatted to three decimal places. This provides a summary of the model's performance in terms of correctly identifying survivors, allowing for easy comparison between different models.
    print(f"--- {name} ---")
    print(f"Precision (survived): {precision:.3f}")
    print(f"Recall (survived):    {recall:.3f}")
    print(f"F1 (survived):        {f1:.3f}\n")

# Full report (precision/recall/F1 for both classes) for the last model above
print(classification_report(y_test, last_pred, target_names=['Died', 'Survived']))

## Step 8: Feature importance check
Finally, you asked the random forest which features it actually leaned on, and checked that against your Step 2 intuition. It ranked fare and age highest, with sex close behind. That roughly lines up with the EDA — sex and class showed the biggest survival gaps early on — though fare edged out sex numerically, likely because fare captures class and family size information in one continuous number, so the model found it slightly more useful than the raw categorical sex split.
- # Feature importances for the Random Forest model, which can help identify which features are most influential in predicting survival. The feature importances are extracted from the fitted Random Forest model and displayed in descending order.
rf_pipe = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', RandomForestClassifier(n_estimators=200, random_state=42))
])
# Fit a Random Forest on the training set to inspect which features it relies on
rf_pipe.fit(X_train, y_train)

# Feature names after preprocessing one-hot encoding expands each categorical column into multiple binary columns
feature_names = rf_pipe.named_steps['preprocessor'].get_feature_names_out()
importances = rf_pipe.named_steps['classifier'].feature_importances_

# Create a pandas Series to hold the feature importances, using the feature names as the index. The Series is then sorted in descending order to display the most important features first. The rounded values of the importances are printed to provide a clear overview of which features have the greatest impact on the model's predictions.
importance_series = pd.Series(importances, index=feature_names).sort_values(ascending=False)
print("Feature importances (Random Forest):")
print(importance_series.round(3))

# Compare against the step 2 EDA: sex and pclass showed the largest
# survival-rate gaps there, so we'd expect them to dominate the importances too