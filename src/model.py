from xgboost import XGBClassifier

def build_xgboost(scale_pos_weight):
    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='auc',
        learning_rate=0.1,
        max_depth=4,
        n_estimators=200,
        scale_pos_weight=scale_pos_weight,
        random_state=42
    )
    return model
