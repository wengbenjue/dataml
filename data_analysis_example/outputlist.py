from data_analysis_example.init_model_ml1 import mlModel

if __name__ == '__main__':
    mlModel.output_list(model_type='lgb', min_lift=1.4,encoder_type='onehot')