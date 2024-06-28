from jump_regressor import predict_skip , get_data , regressor_pipeline
from duration_estimator import predict_duration , estimator_pipeline

df = get_data()



def controller(user_data: dict):
    try:
        model, action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler,df = regressor_pipeline(df)
        skip = predict_skip(model, user_data,action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler,df )
        if int(skip) == 1:
            model, action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler,df = estimator_pipeline(df)
            duration = predict_duration(model,user_data, action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler,df)
            return duration
        else:
            return print('User is not going to skip')
        
    except Exception as e:
        print(e)

    