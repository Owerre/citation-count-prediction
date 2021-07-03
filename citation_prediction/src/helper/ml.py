####################################
# Author: S. A. Owerre
# Date modified: 09/06/2021
# Class: Machine learning
####################################


# Filter warnings
import warnings
warnings.filterwarnings("ignore")

# Data manipulation and visualization
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Model performance metrics
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.metrics import accuracy_score, auc, recall_score
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.metrics import average_precision_score, precision_recall_curve

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
        model: supervised regression model
        X_train: feature matrix of the training set
        y_train: target variable
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
        y_train: target variable
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
        y_pred: training set target variable
        y_true: test set target variable

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
        ax1.set_xlim([-0.1, 9])
        ax1.set_ylim(ylim)
        ax1.hlines(y=0, xmin=-0.1, xmax=9, lw=2, color='k')
        ax1.set_xlabel('Predicted values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs. Predicted values')

        ax2.scatter(y_pred, y_true, color='b')
        ax2.plot([-0.3, 9], [-0.3, 9], color='k')
        ax2.set_xlim([-0.3, 9])
        ax2.set_ylim([-0.3, 9])
        ax2.text(2, 7, r'$R^2 = {},~ RMSE = {}$'.format(str(r2), str(rm)), fontsize=20)
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

################################ Supervised classification ########################################

class SupervisedModels:
    """
    Class for training and testing supervised classification models
    """

    def __init__(self):
        """
        Parameter initialization
        """

    def eval_metrics_cv(self, model, X_train, y_train, cv_fold, scoring = None, model_nm = None):
        """
        Cross-validation on the training set

        Parameters
        ___________
        model: supervised classification model
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold
        scoring: performance metric
        model_nm: name of classifier

        Returns
        _____________
        Performance metrics on the cross-validation training set
        """

        # Fit the training set
        model.fit(X_train, y_train)

        # Compute accuracy on k-fold cross validation
        score = cross_val_score(model, X_train, y_train,cv=cv_fold, scoring = scoring)

        # Make prediction on k-fold cross validation
        y_cv_pred = cross_val_predict(model, X_train, y_train, cv=cv_fold)

        # Make probability prediction on k-fold cross validation
        y_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv_fold, method='predict_proba')[:,1]

        # Print results
        print('{}-Fold cross-validation results for {}'.format(str(cv_fold), str(model_nm)))
        print('-' * 60)
        print('Accuracy (std): %f (%f)' % (score.mean(), score.std()))
        print('AUROC: %f' % (roc_auc_score(y_train, y_pred_proba)))
        print('AUPRC: %f' % (average_precision_score(y_train, y_pred_proba)))
        print('Predicted classes:', np.unique(y_cv_pred))
        print('Confusion matrix:\n', confusion_matrix(y_train, y_cv_pred))
        print('Classification report:\n', classification_report(y_train, y_cv_pred))
        print('-' * 60)


    def plot_auc_ap_svm(self, X_train, y_train, cv_fold = None, class_weight = None):
        """
        Plot of cross-validation AUC and AP for SVM

        Parameters
        ___________
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold

        Returns
        _____________
        matplolib figure of auc vs. hyperparameters
        """
        C_list = [2**x for x in range(-2,9,2)]
        gamma_list = [2**x for x in range(-11,-5,2)]
        auc_list = [pd.Series(0.0, index = range(len(C_list))) for _ in range(len(gamma_list))]
        ap_list = [pd.Series(0.0, index = range(len(C_list))) for _ in range(len(gamma_list))]

        axes_labels = ['2^-2', '2^0', '2^2', '2^4', '2^6', '2^8']
        gamma_labels = ['2^-11', '2^-7', '2^-5']
        plt.rcParams.update({'font.size': 15})
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,6))

        for i, val1 in enumerate(gamma_list):
            for j, val2 in enumerate(C_list):
                model = SVC(C = val2, gamma = val1, probability = True, kernel = 'rbf', 
                            class_weight = class_weight, random_state = 42)
                model.fit(X_train, y_train)
                y_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv_fold,
                                                 method='predict_proba')[:,1]
                auc_list[i][j] = roc_auc_score(y_train, y_pred_proba)
                ap_list[i][j] = average_precision_score(y_train, y_pred_proba)
            auc_list[i].plot(label = "gamma="+str(gamma_labels[i]), marker = "o", linestyle = "-", ax = ax1)
            ap_list[i].plot(label = "gamma="+str(gamma_labels[i]), marker = "o", linestyle = "-", ax = ax2)

        ax1.set_xlabel("C", fontsize = 15)
        ax1.set_ylabel("AUC", fontsize = 15)
        ax1.set_title("{}-Fold Cross-Validation with RBF Kernel SVM".format(cv_fold), fontsize = 15)
        ax1.set_xticklabels(axes_labels)
        ax1.set_xticks(range(len(C_list)))
        ax1.legend(loc = 'best')

        ax2.set_xlabel("C", fontsize = 15)
        ax2.set_ylabel("AP", fontsize = 15)
        ax2.set_title("{}-Fold Cross-Validation with RBF Kernel SVM".format(cv_fold), fontsize = 15)
        ax2.set_xticks(range(len(C_list)))
        ax2.set_xticklabels(axes_labels)
        ax2.legend(loc = 'best')
        plt.show()

    def plot_auc_ap_lr(self, X_train, y_train, cv_fold = None):
        """
        Plot of cross-validation AUC and AP for Logistic regression

        Parameters
        ___________
        X_train: feature matrix of the training set
        y_train: class labels
        cv_fold: number of cross-validation fold

        Returns
        _____________
        matplolib figure of auc vs. hyperparameters
        """
        C_list = [2**x for x in range(-2,9,2)]
        class_wgt_list = [None, 'balanced', {0:1, 1:3}]
        auc_list = [pd.Series(0.0, index = range(len(C_list))) for _ in range(len(class_wgt_list))]
        ap_list = [pd.Series(0.0, index = range(len(C_list))) for _ in range(len(class_wgt_list))]

        axes_labels = ['2^-2', '2^0', '2^2', '2^4', '2^6', '2^8']
        plt.rcParams.update({'font.size': 15})
        fig, (ax1, ax2) = plt.subplots(1,2, figsize = (18,6))

        for i, val1 in enumerate(class_wgt_list):
            for j, val2 in enumerate(C_list):
                model = LogisticRegression(C = val2, class_weight = val1, random_state = 42)
                model.fit(X_train, y_train)
                y_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv_fold, 
                                                 method='predict_proba')[:,1]
                auc_list[i][j] = roc_auc_score(y_train, y_pred_proba)
                ap_list[i][j] = average_precision_score(y_train, y_pred_proba)
            auc_list[i].plot(label = "class_weight="+str(class_wgt_list[i]), marker = "o", linestyle = "-",
                             ax = ax1)
            ap_list[i].plot(label = "class_weight="+str(class_wgt_list[i]), marker = "o", linestyle = "-", 
                            ax = ax2)

        ax1.set_xlabel("C", fontsize = 15)
        ax1.set_ylabel("AUC", fontsize = 15)
        ax1.set_title("{}-Fold Cross-Validation with Logistic Regression".format(cv_fold), fontsize = 15)
        ax1.set_xticklabels(axes_labels)
        ax1.set_xticks(range(len(C_list)))
        ax1.legend(loc = 'best')

        ax2.set_xlabel("C", fontsize = 15)
        ax2.set_ylabel("AP", fontsize = 15)
        ax2.set_title("{}-Fold Cross-Validation with Logistic Regression".format(cv_fold), fontsize = 15)
        ax2.set_xticks(range(len(C_list)))
        ax2.set_xticklabels(axes_labels)
        ax2.legend(loc = 'best')
        plt.show()

    def test_pred(self, model, X_test, y_test, model_nm = None):
        """
        Predictions on the test set

        Parameters
        ___________
        model: trained supervised model
        X_test: feature matrix of the test set
        y_test: test set class labels
        model_nm: name of classifier

        Returns
        _____________
        Performance metrics on the test set
        """
        # Make prediction on the test set
        y_pred = model.predict(X_test)

        # Compute the accuracy of the model
        accuracy = accuracy_score(y_test, y_pred)

        # Predict probability
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        print('Test predictions for {}'.format(str(model_nm)))
        print('-' * 60)
        print('Accuracy:  %f' % (accuracy))
        print('AUROC: %f' % (roc_auc_score(y_test, y_pred_proba)))
        print('AUPRC: %f' % (average_precision_score(y_test, y_pred_proba)))
        print('Predicted classes:', np.unique(y_pred))
        print('Confusion matrix:\n', confusion_matrix(y_test, y_pred))
        print('Classification report:\n', classification_report(y_test, y_pred))
        print('-' * 60)

    def plot_roc_pr_curves(self, model, X_train, y_train, X_test, y_test, cv_fold, color=None, label=None):
        """
        Plot ROC and PR curves for cross-validation and test sets

        Parameters
        ___________
        model: supervised classification model
        X_train: feature matrix of the training set
        y_train: training set class labels
        X_test: feature matrix of the test set
        y_test: test set class labels
        cv_fold: number of k-fold cross-validation
        color: matplotlib color
        label: matplotlib label

        Returns
        _____________
        Matplotlib line plot
        """

        # Fit the model
        model.fit(X_train, y_train)

        ########################## ROC and PR curves for cross-validation set ###########################

        # Make prediction on k-fold cross validation
        y_cv_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv_fold, method="predict_proba")

        # Compute the fpr and tpr for each classifier
        fpr_cv, tpr_cv, thresholds = roc_curve(y_train, y_cv_pred_proba[:, 1])

        # Compute the precisions and recalls for the classifier
        precisions_cv, recalls_cv, thresholds = precision_recall_curve(y_train, y_cv_pred_proba[:, 1])

        # Compute the area under the ROC curve for each classifier
        area_auc_cv = roc_auc_score(y_train, y_cv_pred_proba[:, 1])

        # Compute the area under the PR curve for the classifier
        area_prc_cv = auc(recalls_cv, precisions_cv)

        # ROC curve
        # plt.rcParams.update({'font.size': 12})
        plt.subplot(221)
        plt.plot(fpr_cv, tpr_cv, color=color, label=(label) % area_auc_cv)
        plt.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title('ROC Curve for the Cross-Validation Training Set')
        plt.legend(loc='best')

        # PR curve
        plt.subplot(223)
        plt.plot(recalls_cv, precisions_cv, color=color, label=(label) % area_prc_cv)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for the {}-Fold Cross-Validation Training Set'.format(cv_fold))
        plt.legend(loc='best')

        ############################## ROC and PR curves for Test set #####################################
        # Predict probability
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute the fpr and tpr for each classifier
        fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)

        # Compute the precisions and recalls for the classifier
        precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

        # Compute the area under the ROC curve for each classifier
        area_auc = roc_auc_score(y_test, y_pred_proba)

        # Compute the area under the PR curve for the classifier
        area_prc = auc(recalls, precisions)

        # ROC curve
        # plt.rcParams.update({'font.size': 12})
        plt.subplot(222)
        plt.plot(fpr, tpr, color=color, label=(label) % area_auc)
        plt.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)
        plt.axis([0, 1, 0, 1])
        plt.xlabel('False positive rate (FPR)')
        plt.ylabel('True positive rate (TPR)')
        plt.title('ROC Curve for the Test Set')
        plt.legend(loc='best')

        # PR curve
        plt.subplot(224)
        plt.plot(recalls, precisions, color=color, label=(label) % area_prc)
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('PR Curve for the Test Set')
        plt.legend(loc='best')

    def plot_aucroc_aucpr(self, model, X_train, y_train, X_test, y_test, cv_fold,  marker= None,
                          color = None, label = None):
        """
        Plot AUC-ROC  and AUC-PR curves for cross-validation vs. test sets

        Parameters
        ___________
        model: supervised classification model
        X_train: feature matrix of the training set
        y_train: training set class labels
        X_test: feature matrix of the test set
        y_test: test set class labels
        cv_fold: number of k-fold cross-validation
        color: matplotlib color
        marker: matplotlib marker

        Returns
        _____________
        Matplotlib line plot
        """
        # Fit the model
        model.fit(X_train, y_train)

        ############################## AUC-ROC  and AUC-PR for Test set #####################################

        # Predict probability on the test set
        y_pred_proba = model.predict_proba(X_test)[:, 1]

        # Compute the precisions and recalls of the test set
        test_precisions, test_recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)

        # Compute the area under the ROC curve of the test set
        area_auc_test = roc_auc_score(y_test, y_pred_proba)

        # Compute the area under the PR curve on the test set
        area_prc_test = auc(test_recalls, test_precisions)

        ########################### AUC-ROC  and AUC-PR cross-validation training set#########################

        # Make prediction on the k-fold cross-validation set
        y_cv_pred_proba = cross_val_predict(model, X_train, y_train, cv=cv_fold, method="predict_proba")

        # Compute the precisions and recalls of the cross-validation set
        cv_precisions, cv_recalls, thresholds = precision_recall_curve(y_train, y_cv_pred_proba[:, 1])

        # Compute the area under the ROC curve of the cross-validation set
        area_auc_cv = roc_auc_score(y_train, y_cv_pred_proba[:, 1])

        # Compute the area under the PR curve of the cross-validation set
        area_prc_cv = auc(cv_recalls, cv_precisions)

        ############################ Plot #############################################
        # AUC-ROC 
        plt.subplot(121)
        plt.plot([area_auc_cv], [area_auc_test], color = color, marker = marker, label = label)
        plt.plot([0.70, 1], [0.70, 1], 'k--', linewidth = 0.5)
        plt.axis([0.70,1.01,0.70,1.01])
        plt.xticks(np.arange(0.70,1.02,0.05))
        plt.yticks(np.arange(0.70,1.02,0.05))
        plt.xlabel('Cross-validation set results')
        plt.ylabel('Test set results')
        plt.title('AUC-ROC for Cross-Validation vs. Test Sets')
        plt.legend(loc='best')

        # AUC-PR
        plt.subplot(122)
        plt.plot([area_prc_cv], [area_prc_test], color = color, marker = marker, label = label)
        plt.plot([0, 1], [0, 1], 'k--', linewidth = 0.5)
        plt.axis([0.3,1.01,0.3,1.01])
        plt.xticks(np.arange(0.3,1.05,0.1))
        plt.yticks(np.arange(0.3,1.05,0.1))
        plt.xlabel('Cross-validation set results')
        plt.ylabel('Test set results')
        plt.title('AUC-PR for Cross-Validation vs. Test Sets')
        plt.legend(loc='best')