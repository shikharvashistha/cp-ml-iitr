import pandas as pd
import seaborn as sns
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import Ridge, Lasso, ElasticNet, BayesianRidge, HuberRegressor, QuantileRegressor, TheilSenRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
import lightgbm as lgb
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel
from sklearn.isotonic import IsotonicRegression


labelled_dataset_csv = pd.read_csv('dataset/train.csv')

def analysis():
    # plot graph
    plt.plot(labelled_dataset_csv['yoe'],labelled_dataset_csv['salary'])
    plt.xlabel('Years of Experience')
    plt.ylabel('Salary')
    plt.savefig('graphs/analysis.png')

    # plot violin plot
    sns.violinplot(x=labelled_dataset_csv['yoe'])
    plt.savefig('graphs/violin_plot.png')


def data():
    X_train, X_test, y_train, y_test = train_test_split(labelled_dataset_csv['yoe'],labelled_dataset_csv['salary'],test_size=0.2,random_state=0)
    X_train = X_train.values.reshape(-1,1)
    X_test = X_test.values.reshape(-1,1)
    y_train = y_train.values.reshape(-1,1)
    y_test = y_test.values.reshape(-1,1)
    return X_train,y_train,X_test,y_test

def ann():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # train model
    model = MLPRegressor(hidden_layer_sizes=(100,100,100),activation='relu',solver='adam',max_iter=50000)
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # MSE calculation - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/ann.png')

def ridge_regression():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    # train model
    model = Ridge()
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    # print mse in graph
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/ridge_regression.png')

def lasso_regression():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    # train model
    model = Lasso()
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/lasso_regression.png')

def elastic_net():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    # train model
    model = ElasticNet(alpha=0.1, l1_ratio=0.5)
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/elastic_net.png')

def svr():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # train model
    model = SVR(kernel='linear', C=1.0, epsilon=0.1)
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/svr.png')

def decision_tree_regression():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    # train model
    model = DecisionTreeRegressor(criterion='friedman_mse',  # Splitting criterion ('mse','friedman_mse','mae')
    splitter='best',   # Strategy for choosing splits ('best' or 'random')
    max_depth=None,    # Maximum depth of the tree (None means nodes are expanded until they contain less than min_samples_split samples)
    min_samples_split=2,  # Minimum number of samples required to split an internal node
    min_samples_leaf=1,   # Minimum number of samples required to be at a leaf node
    max_features=None,   # Number of features to consider when looking for the best split
    random_state=42)
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/decision_tree_regression.png')

def random_forest_regression():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # train model
    model = RandomForestRegressor(
    n_estimators=100,     # Number of trees in the forest
    criterion='friedman_mse',      # Splitting criterion ('mse','friedman_mse','mae')
    max_depth=None,        # Maximum depth of the trees (None means nodes are expanded until they contain less than min_samples_split samples)
    min_samples_split=2,   # Minimum number of samples required to split an internal node
    min_samples_leaf=1,    # Minimum number of samples required to be at a leaf node
    max_features='sqrt',  # Number of features to consider when looking for the best split ('auto' means sqrt(n_features))
    random_state=42        # Seed for the random number generator
    )
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/random_forest_regression.png')

def gradient_boosting_regression():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # train model
    model = GradientBoostingRegressor(n_estimators=500,max_depth=4,min_samples_split=5,learning_rate=0.01,loss='squared_error')
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/gradient_boosting_regression.png')

def xgboost():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    # train model
    model = xgb.XGBRegressor(objective='reg:squarederror', n_estimators=1000, seed=123)
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/xgboost.png')

def lightgbm():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    # train model
    model = lgb.LGBMRegressor(min_data_in_leaf=2)
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/lightgbm.png')

def knn():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    # train model
    model = KNeighborsRegressor(n_neighbors=5)
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/knn.png')

def gaussian_processes():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    # Specify the kernel for the Gaussian process
    kernel = DotProduct() + WhiteKernel()

    # Create an instance of GaussianProcessRegressor with custom hyperparameters
    model = GaussianProcessRegressor(
    kernel=kernel,        # The kernel specifying the covariance function
    n_restarts_optimizer=10,  # Number of restarts for optimizing the kernel's hyperparameters
    random_state=42,      # Seed for random number generation for reproducibility
    optimizer='fmin_l_bfgs_b',  # Explicitly set the optimizer
    )

    # train model
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/gaussian_processes.png')

def bayesian_regression():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # train model
    model = BayesianRidge()
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/bayesian_regression.png')

def isotonic_regression():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    X_train = X_train.ravel()
    X_test = X_test.ravel()
    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # train model
    model = IsotonicRegression(out_of_bounds='clip')
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/isotonic_regression.png')

def huber_regression():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # train model
    model = HuberRegressor(
    epsilon=1.35,  # The parameter controlling the threshold for the outliers
    max_iter=100,  # Maximum number of iterations for optimization
    alpha=0.0001   # Regularization strength
    )
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/huber_regression.png')

def quantile_regression():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # train model
    model = QuantileRegressor(
    quantile=0.5,  # Quantile to be estimated
    alpha=0.95,    # Confidence level for the confidence interval around the prediction
    solver='revised simplex', # Solver for the linear optimization problem ('revised simplex','interior-point')
    )
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/quantile_regression.png')

def theil_sen_regression():
    # split dataset into train and test
    X_train, y_train, X_test, y_test = data()

    y_train = y_train.ravel()
    y_test = y_test.ravel()

    # train model
    model = TheilSenRegressor(
        fit_intercept=True,  # Whether to calculate the intercept for this model
        random_state=42      # Seed for random number generation for reproducibility
    )
    model.fit(X_train,y_train)

    # predict labels
    y_pred = model.predict(X_test)

    # calculate mse - normalise it to get accuracy
    mse = mean_squared_error(y_test,y_pred)
    mse = mse/10000000
    print("MSE: ",mse)

    # plot graph
    plt.plot(y_test,y_pred)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.text(0.95, 0.95, f'MSE: {mse:.2f}', ha='right', va='top', transform=plt.gca().transAxes, bbox=dict(facecolor='white', alpha=0.5))
    plt.savefig('graphs/theil_sen_regression.png')

# augument data using polynomial features and noise
num_augmented_samples = 3000
def augument():
    data = pd.read_csv('dataset/train.csv')
    augmented_data = []

    for _, row in data.iterrows():
        original_yoe = row['yoe']
        original_salary = row['salary']

        # Add noise to the original data
        for _ in range(num_augmented_samples):
            # yoe - upto 1 decimal place
            noise_yoe = original_yoe + np.random.uniform(-0.1, 0.1)
            noise_salary = original_salary + np.random.uniform(-500, 500)

            augmented_data.append({'yoe': noise_yoe.round(1), 'salary': noise_salary.round(2)})

        # Generate polynomial features
        for degree in range(2, 4):  # You can adjust the degree of the polynomial
            poly_yoe = original_yoe ** degree
            augmented_data.append({'yoe': poly_yoe, 'salary': original_salary})

    augmented_data = pd.DataFrame(augmented_data)

    # Save the augmented data
    augmented_data.to_csv('dataset/augmented_data.csv', index=False)


if __name__ == "__main__":

    print("Enter choice: ")
    print("0. Augument data")
    print("1. Analysis")
    print("2. ANN")
    print("3. Ridge Regression")
    print("4. Lasso Regression")
    print("5. Elastic Net")
    print("6. SVR")
    print("7. Decision Tree Regression")
    print("8. Random Forest Regression")
    print("9. Gradient Boosting Regression")
    print("10. XGBoost")
    print("11. LightGBM")
    print("12. KNN")
    print("13. Gaussian Processes")
    print("14. Bayesian Regression")
    print("15. Isotonic Regression")
    print("16. Huber Regression")
    print("17. Quantile Regression")
    print("18. Theil Sen Regression")

    print('')

    choice = int(input())

    if choice == 0:
        augument()
    elif choice == 1:
        analysis()
    elif choice == 0:
        format()
    elif choice == 2:
        ann()
    elif choice == 3:
        ridge_regression()
    elif choice == 4:
        lasso_regression()
    elif choice == 5:
        elastic_net()
    elif choice == 6:
        svr()
    elif choice == 7:
        decision_tree_regression()
    elif choice == 8:
        random_forest_regression()
    elif choice == 9:
        gradient_boosting_regression()
    elif choice == 10:
        xgboost()
    elif choice == 11:
        lightgbm()
    elif choice == 12:
        knn()
    elif choice == 13:
        gaussian_processes()
    elif choice == 14:
        bayesian_regression()
    elif choice == 15:
        isotonic_regression()
    elif choice == 16:
        huber_regression()
    elif choice == 17:
        quantile_regression()
    elif choice == 18:
        theil_sen_regression()
    else:
        print("Wrong choice")