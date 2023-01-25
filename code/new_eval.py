from sklearn import metrics as sklearn_metrics

def temp_eval(labels, preds):
    return {

         "f1_micro": sklearn_metrics.f1_score(labels, preds, average="micro", zero_division=0),
         "f1_macro": sklearn_metrics.f1_score(labels, preds, average="macro", zero_division=0),
         "acc": sklearn_metrics.accuracy_score(labels, preds),
        }