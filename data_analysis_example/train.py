from data_analysis_example.init_model_ml1 import mlModel

if __name__ == '__main__':
    # mlModel.train()
    # mlModel.train(model_type='lr',encoder_type='onehot')
    # mlModel.train(model_type='lgb',encoder_type='onehot')
    # mlModel.train(model_type='decision_tree',encoder_type='onehot')

    # mlModel.train(model_type='lgb',encoder_type='woe')
    # mlModel.train(model_type='lr',encoder_type='woe')
    # mlModel.train(model_type='lr',encoder_type='woe')
    # mlModel.outtime(model_type='lr',encoder_type='woe')
    mlModel.train(model_type='catboost', encoder_type=None)
    mlModel.outtime(model_type='catboost', encoder_type=None, need_load_model=False)
    psi = mlModel.oot_model_psi(score_name='y_prob', p=10, model_type='catboost')