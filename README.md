# Report
## Step 1: Imported all the required libraries

## Step 2: Reading the csv file(i.e, the dataset)

## Step 3: Used the info() function 

The Pandas dataframe info() method provides information on the number of row-entries in the dataframe and the number of columns in the dataframe. The count of non-null entries per column, the data type of each column, and the memory usage of the dataframe is also provided.

## Step 4: Checked the total null values in the data

## Step 5: Split the data into dependent and independent features

## Step 6: Used StandardScaler() function 

The StandardScaler standardizes the features by making the mean equal to zero and the variance equal to one
## Step 7: Split the data into training and testing data
## Step 8: Initialized a Logistic Reggression variable with hyperparameters

The parameter C specifies regularization strength. Regularization implies penalizing the model for overfitting. C=1.0 is the default value for LogisticRegressor in the sklearn library.

The class_weight=’balanced’ method provides weights to the classes. If unspecified, the default class_weight is = 1. Class weight = ‘balanced’ assigns class weights by using the formula (n_samples/(n_classes*np.bin_count(y))). e.g. if n_samples =100, n_classes=2 and there are 50 samples belonging to each of the 0 and 1 classes, class_weight = 100/(2*50) = 1

dual = False is preferable when n_samples > n_features. dual formulation is implemented only for the L2 regularizer with liblinear solver.

N.B. Liblinear solver utilizes the coordinate-descent algorithm instead of the gradient descent algorithms to find the optimal parameters for the logistic regression model. E.g. in the gradient descent algorithms, we optimize all the parameters at once. While coordinate descent optimizes only one parameter at a time. In coordinate descent, we first initialize the parameter vector (theta = [theta0, theta1 …….. thetan]). In the kth iteration, only thetaik is updated while (theta0k… thetai-1k and thetai+1k-1…. thetank-1) are fixed.

fit_intercept = True The default value is True. Specifies if a constant should be added to the decision function.

intercept_scaling = 1 The default value is 1. Is applicable only when the solver is liblinear and fit_intercept = True. [X] becomes [X, intercept_scaling]. A synthetic feature with constant value = intercept_scaling is appended to [X]. The intercept becomes, intercept scaling * synthetic feature weight. Synthetic feature weight is modified by L1/L2 regularizations. To lessen the effect of regularization on synthetic feature weights, high intercept_scaling value must be chosen.

max_iter = 100 (default). A maximum number of iterations is taken for the solvers to converge.

multi_class = ‘ovr’, ‘multinomial’ or auto(default). auto selects ‘ovr’ i.e. binary problem if the data is binary or if the solver is liblinear. Otherwise auto selects multinomial which minimises the multinomial loss function even when the data is binary.

n_jobs (default = None). A number of CPU cores are utilized when parallelizing computations for multi_class=’ovr’. None means 1 core is used. -1 means all cores are used. Ignored when the solver is set to liblinear.

penalty: specify the penalty norm (default = L2).

random_state = set random state so that the same results are returned every time the model is run.

solver = the choice of the optimization algorithm (default = ‘lbfgs’)

tol = Tolerance for stopping criteria (default = 1e-4)

verbose = 0 (for suppressing information during the running of the algorithm)

warm_start = (default = False). when set to True, use the solution from the previous step as the initialization for the present step. This is not applicable for the liblinear solver.

## Step 9: Fitted the training data to make our model

## Step 10: Tested the accuracy of our model

## Conclusion

Developed a logistic regression model for heart disease prediction using a dataset from the UCI repository. Focused on gaining an in-depth understanding of the hyperparameters, libraries and code used when defining a logistic regression model through the scikit-learn library.
