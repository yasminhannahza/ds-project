import xgboost as xgb

def load_model_future():
    model = xgb.Booster()
    model.load_model('future_xgboost_model.json.json')
    return model

def load_model_past():
    model = xgb.Booster()
    model.load_model('past_xgboost_model.json.json')
    return model
