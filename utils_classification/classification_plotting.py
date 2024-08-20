import numpy as np
from sklearn .metrics import roc_auc_score, roc_curve
from plotly import graph_objects as go


def compute_and_plot_ROC_and_AUC(data_ground_truth: np.ndarray, data_prediction: np.ndarray):

    auc = np.round(roc_auc_score(data_ground_truth, data_prediction), 3)
    print("Auc for our sample data is {}".format(auc))

    false_pos_rate, true_pos_rate, _ = roc_curve(data_ground_truth,  data_prediction)

    trace = go.Scatter(x=false_pos_rate, y=true_pos_rate, mode='lines', name='AUC = %0.2f' % auc,
                    line=dict(color='darkorange', width=2))
    reference_line = go.Scatter(x=[0,1], y=[0,1], mode='lines', name='Reference Line',
                                line=dict(color='navy', width=2, dash='dash'))
    fig = go.Figure(data=[trace, reference_line])
    fig.update_layout(title='Interactive ROC Curve',
                    xaxis_title='False Positive Rate',
                    yaxis_title='True Positive Rate')
    fig.show()