# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Helps with importing functions from different directory
import os
import sys
BASE_PATH = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_PATH)

# Model performance
from sklearn.model_selection import cross_val_predict

# import custom class
from helper import log_transfxn as cf 


class RegressionModels:
    """
    Class for training and testing supervised regression models
    """

    def __init__(self):
        """
        Parameter initialization
        """

    def eval_metric_cv(self, model, X_train, y_train, cv_fold, model_nm = None):
        """
        Cross-validation on the training set

        Parameters
        ___________
        model: supervised classification model
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold

        Returns
        _____________
        Performance metrics on the cross-validation training set
        """

        # Fit the training set
        model.fit(X_train, y_train)

        # Make prediction on k-fold cross validation set
        y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv_fold)

        # Print results
        print('{}-Fold cross-validation results for {}'.format(str(cv_fold), str(model_nm)))
        print('-' * 45)
        print(self.error_metrics(y_train, y_pred_cv))
        print('-' * 45)
    
    def plot_mae_rsme_svr(self, X_train, y_train, cv_fold):
        """
        Plot of cross-validation MAE and RMSE for SVR

        Parameters
        ___________
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold

        Returns
        _____________
        matplolib figure of MAE & RMSE
        """
        C_list = [2**x for x in range(-2,11,2)]
        gamma_list = [2**x for x in range(-7,-1,2)]
        mae_list = [pd.Series(0.0, index = range(len(C_list))) for _ in range(len(gamma_list))]
        rmse_list = [pd.Series(0.0, index = range(len(C_list))) for _ in range(len(gamma_list))]

        axes_labels = ['2^-2', '2^0', '2^2', '2^4', '2^6', '2^8', '2^10']
        gamma_labels = ['2^-7', '2^-5', '2^-3']
        plt.rcParams.update({'font.size': 15})
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,6))

        for i, val1 in enumerate(gamma_list):
            for j, val2 in enumerate(C_list):
                model = SVR(C = val2, gamma = val1, kernel = 'rbf')
                model.fit(X_train, y_train)
                y_pred_cv = cross_val_predict(model, X_train, y_train, cv=cv_fold)
                mae_list[i][j] = self.mae(y_train, y_pred_cv)
                rmse_list[i][j] = self.rmse(y_train, y_pred_cv)
            mae_list[i].plot(label = "gamma="+str(gamma_labels[i]), marker = "o", linestyle = "-", ax = ax1)
            rmse_list[i].plot(label = "gamma="+str(gamma_labels[i]), marker = "o", linestyle = "-", ax = ax2)

        ax1.set_xlabel("C", fontsize = 15)
        ax1.set_ylabel("MAE", fontsize = 15)
        ax1.set_title("{}-Fold Cross-Validation with RBF kernel SVR".format(cv_fold), fontsize = 15)
        ax1.set_xticklabels(axes_labels)
        ax1.set_xticks(range(len(C_list)))
        ax1.legend(loc = 'best')

        ax2.set_xlabel("C", fontsize = 15)
        ax2.set_ylabel("RSME", fontsize = 15)
        ax2.set_title("{}-Fold Cross-Validation with RBF kernel SVR".format(cv_fold), fontsize = 15)
        ax2.set_xticks(range(len(C_list)))
        ax2.set_xticklabels(axes_labels)
        ax2.legend(loc = 'best')
        plt.show()
        
    def eval_metric_test(self, y_pred, y_true, model_nm = None):
        """
        Predictions on the test set

        Parameters
        ___________
        y_pred: training set class labels
        y_true: test set class labels

        Returns
        _____________
        Performance metrics on the test set
        """
        # Print results
        print('Test prediction results for {}'.format(model_nm))
        print('-' * 45)
        print(self.error_metrics(y_true, y_pred))
        print('-' * 45)
        
    def diagnostic_plot(self, y_pred, y_true, ylim = None):
        """
        Diagnostic plot
        
        Parameters
        ___________
        y_pred: predicted labels
        y_true: true labels

        Returns
        _____________
        Matplolib figure
        """
        # Compute residual and metrics
        residual = (y_true - y_pred)
        r2 = np.round(self.r_squared(y_true, y_pred), 3)
        rm = np.round(self.rmse(y_true, y_pred), 3)
        
        # Plot figures
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))
    
        ax1.scatter(y_pred, residual, color ='b')
        ax1.set_xlim([-0.1, 14])
        ax1.set_ylim(ylim)
        ax1.hlines(y=0, xmin=-0.1, xmax=14, lw=2, color='k')
        ax1.set_xlabel('Predicted values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs. Predicted values')

        ax2.scatter(y_pred, y_true, color='b')
        ax2.plot([-0.3, 14], [-0.3, 14], color='k')
        ax2.set_xlim([-0.3, 14])
        ax2.set_ylim([-0.3, 14])
        ax2.text(2, 12, r'$R^2 = {},~ RMSE = {}$'.format(str(r2), str(rm)), fontsize=20)
        ax2.set_xlabel('Predicted values')
        ax2.set_ylabel('True values')
        ax2.set_title('True values vs. Predicted values')
    
    def error_metrics(self, y_true, y_pred):
        """
        Print out error metrics
        """
        r2 = self.r_squared(y_true, y_pred)
        mae = self.mae(y_true, y_pred)
        rmse = self.rmse(y_true, y_pred)

        result = {'MAE = {}'.format(np.round(mae,3)),
                  'RMSE = {}'.format(np.round(rmse,3)),
                  'R^2 = {}'.format(np.round(r2,3))}
        return result

    def mae(self, y_test, y_pred):
        """
        Mean absolute error
        
        Parameters
        ___________
        y_test: test set label
        y_pred: prediction label

        Returns
        _____________
        Mean absolute error
        """
        mae = np.mean(np.abs((y_test - y_pred)))
        return mae


    def rmse(self, y_test, y_pred):
        """
        Root mean squared error
        
        Parameters
        ___________
        y_test: test set label
        y_pred: prediction label

        Returns
        _____________
        Root mean squared error
        """
        rmse = np.sqrt(np.mean((y_test - y_pred)**2))
        return rmse


    def r_squared(self, y_test, y_pred):
        """
        r-squared (coefficient of determination)
        
        Parameters
        ___________
        y_test: test set label
        y_pred: prediction label

        Returns
        _____________
        r-squared
        """
        mse = np.mean((y_test - y_pred)**2)  # mean squared error
        var = np.mean((y_test - np.mean(y_test))**2)  # sample variance
        r_squared = 1 - mse / var
        return r_squared