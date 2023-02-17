"""
Refer to: https://en.wikipedia.org/wiki/Precision_and_recall#Definition_(classification_context)
"""

import  numpy as np
def accuracy(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    #acc = np.sum(np.equal(y_true, y_pred)) / len(y_true)

    tp=np.sum(np.logical_and(y_pred==1.0,y_true == 1.0))
    fp=np.sum(np.logical_and(y_pred==1.0,y_true == 0.0))
    fn=np.sum(np.logical_and(y_pred==0.0,y_true == 1.0))
    tn=np.sum(np.logical_and(y_pred==0.0,y_true == 0.0))
    acc=(tp+tn)/(tp+fp+tn+fn)
    return acc
    

def precision_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    tp=np.sum(np.logical_and(y_pred==1.0,y_true==1.0))
    fp=np.sum(np.logical_and(y_pred==1.0,y_true==0.0))
    return (tp/(tp+fp))
    # todo: implement


def recall_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """

    tp=np.sum(np.logical_and(y_pred==1.0,y_true==1.0))
    fn=np.sum(np.logical_and(y_pred==0.0,y_true==1.0))
    return (tp/(tp+fn))
    # todo: implement



def f1_score(y_true, y_pred):
    """

    :param y_true:
    :param y_pred:
    :return:
    """
    # todo: implement
    recall=recall_score(y_true,y_pred)
    prec=precision_score(y_true,y_pred)
    f1=(2*prec*recall)/(recall+prec)
    return f1
