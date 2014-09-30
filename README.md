Titanic-Machine-Learning-from-Disaster
======================================

My Solutions to a Kaggle Competition:

1. GAP model:
    <i>
        Considered three variables: Gender, Age, and Passenger class, as the explanatory variables;
        Split the passengers into different groups respectively according to each of the three variables;
        Use the percentage of the passengers survived the Titanic sinking in each of the groups as the survival probabilities of a given passenger in the test dataset;
        Average the three probabilities in each group, and if it is greater than 50%, then predict the passenger survived.
    </i>
    The accuracy is: around 81% my own on cross-validation set, and 78.5% on Kaggle.

2. Classification Using Six Commonly Used Classifiers and Combining Them With Majority Vote:
    <i>
        Used six commonly used classification algorithms: logistic regression, decision trees, random forest, k-nearest neighbors, linear discriminant analysis, support vector machine, and combining all of them with majority vote.
    </i>


