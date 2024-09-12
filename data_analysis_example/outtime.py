from data_analysis_example.init_model_ml1 import mlModel

if __name__ == '__main__':
    # mlModel.outtime(model_type='lr')
    # mlModel.outtime(model_type='lgb',encoder_type='onehot')
    mlModel.outtime(model_type='lgb',encoder_type=None,need_load_model=True,need_imbance_process=True)
    # psi = mlModel.oot_model_psi(score_name='score', p=10, model_type='lgb')
    psi = mlModel.oot_model_psi(score_name='y_prob', p=10, model_type='lgb',encoder_type=None)
    print(psi)