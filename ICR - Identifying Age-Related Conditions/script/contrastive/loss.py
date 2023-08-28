import numpy as np

def lgb_augment_metric(y_pred, y_true):
    y_true = y_true.get_label()
    y_pred = logistics(y_pred)
    return 'balanced_log_loss', competition_log_loss(y_true, y_pred), False

def lgb_metric(y_pred, y_true):
    y_true = y_true.get_label()
    return 'balanced_log_loss', competition_log_loss(y_true, y_pred), False

def competition_log_loss(y_true, y_pred):    
    #  _ _ _ The competition log loss - class weighted _ _ _
    # The Equation on the Evaluation page is the competition log loss
    # provided w_0=1 and w_1=1.
    # That is, the weights shown in the equation
    # are in addition to class balancing.
    # For this case:
    # _ A constant y_pred = 0.5 will give loss = 0.69 for any class ratio.
    # _ Predicting the observed training ratio, p_1 = 0.17504
    #   will give loss ~ 0.96 for any class ratio.
    #   This is confirmed by the score for this notebook:
    #   https://www.kaggle.com/code/avtandyl/simple-baseline-mean 
    # y_true: correct labels 0, 1
    # y_pred: predicted probabilities of class=1
    # Implements the Evaluation equation with w_0 = w_1 = 1.
    # Calculate the number of observations for each class
    N_0 = np.sum(1 - y_true)
    N_1 = np.sum(y_true)
    # Calculate the predicted probabilities for each class
    p_1 = np.clip(y_pred, 1e-15, 1 - 1e-15)
    p_0 = 1 - p_1
    # Calculate the average log loss for each class
    log_loss_0 = -np.sum((1 - y_true) * np.log(p_0)) / N_0
    log_loss_1 = -np.sum(y_true * np.log(p_1)) / N_1
    # return the (not further weighted) average of the averages
    return (log_loss_0 + log_loss_1)/2


def table_augmentation_logloss(real_target_mask, y_pred, data):
    y_true = np.array(data.get_label())
    y_pred = np.array(y_pred)

    grad, hess = np.zeros((len(y_true))), np.zeros((len(y_true)))
    feat_grad, feat_hess = logloss_derivative(
        y_true[~real_target_mask], y_pred[~real_target_mask]
    )
    grad[~real_target_mask] = feat_grad/20
    hess[~real_target_mask] = feat_hess

    target_grad, target_hess = logloss_derivative(
        y_true[real_target_mask], y_pred[real_target_mask]
    )

    grad[real_target_mask] = target_grad
    hess[real_target_mask] = target_hess
    return grad, hess

def rmse_derivative(y_true, y_pred):
    error = (y_pred-y_true) * 1/20 #(Nreal/Nsim) magic coefficient?

    #1st derivative of loss function
    grad = 2. * error

    #2nd derivative of loss function
    hess = 2.
    
    return grad, hess

def logloss_derivative(y_true, y_pred):
    preds = logistics(y_pred)

    grad = (preds - y_true)

    hess = (preds * (1.0 - preds))

    return grad, hess

def logistics(x):
    return 1.0 / (1.0 + np.exp(-x))