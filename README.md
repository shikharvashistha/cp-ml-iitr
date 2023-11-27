
# Regression Analysis with Machine Learning Models

This repository contains Python code for performing regression analysis on a dataset using various machine learning models. The code includes implementations for artificial neural networks (ANN), ridge regression, lasso regression, elastic net, support vector regression (SVR), decision tree regression, random forest regression, gradient boosting regression, XGBoost, LightGBM, k-nearest neighbors (KNN), Gaussian processes, Bayesian regression, isotonic regression, Huber regression, quantile regression, and Theil-Sen regression.

## Requirements

- Python 3.x
- Required Python packages: pandas, seaborn, plotly, numpy, xgboost, scikit-learn, matplotlib, pickle, lightgbm

## Dataset

The code assumes that the dataset is stored in a CSV file named `train.csv`. Ensure that the dataset is available before running the code.

## Usage

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/your_project.git
   ```

2. Install the required packages:

   ```bash
   pip3 install -r requirements.txt
   ```

3. Run the analysis:

   ```bash
   python3 main.py
   ```

4. Follow the on-screen instructions to choose the analysis type.

## Analysis Types

1. **Analysis:** Generates graphs for the dataset, including a line plot and a violin plot.

2. **ANN (Artificial Neural Network):** Trains an MLPRegressor model and evaluates its performance.

3. **Ridge Regression:** Trains a Ridge regression model and evaluates its performance.

4. **Lasso Regression:** Trains a Lasso regression model and evaluates its performance.

5. **Elastic Net:** Trains an Elastic Net regression model and evaluates its performance.

6. **SVR (Support Vector Regression):** Trains an SVR model and evaluates its performance. 

7. **Decision Tree Regression:** Trains a decision tree regression model and evaluates its performance.

8. **Random Forest Regression:** Trains a random forest regression model and evaluates its performance.

9. **Gradient Boosting Regression:** Trains a gradient boosting regression model and evaluates its performance.

10. **XGBoost:** Trains an XGBoost regression model and evaluates its performance.

11. **LightGBM:** Trains a LightGBM regression model and evaluates its performance.

12. **KNN (k-nearest neighbors):** Trains a KNN regression model and evaluates its performance.

13. **Gaussian Processes:** Trains a Gaussian Process regression model and evaluates its performance.

14. **Bayesian Regression:** Trains a Bayesian Ridge regression model and evaluates its performance.

15. **Isotonic Regression:** Trains an isotonic regression model and evaluates its performance.

16. **Huber Regression:** Trains a Huber regression model and evaluates its performance.

17. **Quantile Regression:** Trains a quantile regression model and evaluates its performance.

18. **Theil-Sen Regression:** Trains a Theil-Sen regression model and evaluates its performance.

Choose the desired analysis type by entering the corresponding number when prompted.

## Graphs

Graphs generated during the analysis are saved in the `graphs` directory with filenames corresponding to the analysis type (e.g., `graphs/ann.png`, `graphs/ridge_regression.png`, etc.).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

Feel free to customize this template based on your specific needs. Include additional information or sections as necessary.