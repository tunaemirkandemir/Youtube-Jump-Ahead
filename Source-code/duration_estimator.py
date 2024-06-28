
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder , StandardScaler
import torch

import torch.nn as nn
import torch.optim as optim
from jump_regressor import  get_data,create_sequences,prepare_data,evaluate

def pre_process_duration(df, fit =True):

    df.sort_values(by=['User_ID', 'Timestamp'], inplace=True)

    action_encoder = OneHotEncoder(handle_unknown='ignore')
    videoID_encoder = OneHotEncoder(handle_unknown='ignore')
    topic_encoder = OneHotEncoder(handle_unknown='ignore')
    tag_encoder = OneHotEncoder(handle_unknown='ignore')
    scaler = StandardScaler()
    
    if fit:
        # Fit encoders
        action_encoded = action_encoder.fit_transform(df[['Action']]).toarray()
        videoID_encoded = videoID_encoder.fit_transform(df[['Video_ID']]).toarray()
        topics_encoded = topic_encoder.fit_transform(df['Metadata'].apply(lambda x: x.get('topic') if isinstance(x, dict) else '').values.reshape(-1, 1)).toarray()
        tags_combined = df['Metadata'].apply(lambda x: ' '.join(x['tags']) if isinstance(x, dict) else '')
        tags_encoded = tag_encoder.fit_transform(tags_combined.values.reshape(-1, 1)).toarray()
        
        # Fit scaler on numerical features
        numerical_features = df[['Duration', 'Engagement']].values
        scaled_numerical = scaler.fit_transform(numerical_features)
    else:
        # Transform only
        action_encoded = action_encoder.transform(df[['Action']]).toarray()
        videoID_encoded = videoID_encoder.transform(df[['Video_ID']]).toarray()
        topics_encoded = topic_encoder.transform(df['Metadata'].apply(lambda x: x.get('topic') if isinstance(x, dict) else '').values.reshape(-1, 1)).toarray()
        tags_combined = df['Metadata'].apply(lambda x: ' '.join(x['tags']) if isinstance(x, dict) else '')
        tags_encoded = tag_encoder.transform(tags_combined.values.reshape(-1, 1)).toarray()

        numerical_features = df[['Duration', 'Engagement']].values
        scaled_numerical = scaler.transform(numerical_features)

    model_features = np.hstack((action_encoded, topics_encoded, videoID_encoded, tags_encoded, scaled_numerical))

    target = df['Duration_Skipped'].values

    grouped_features = list(df.groupby('User_ID').indices.values())
    

    return  model_features, target, grouped_features, action_encoder, videoID_encoder, topic_encoder, tag_encoder, scaler


class SkipDurationPredictor(nn.Module):

    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(SkipDurationPredictor, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        return out
    
def prepare_estimator(model_features,train_dataloader):

    input_dim = model_features.shape[1]
    hidden_dim = 64
    num_layers = 2
    output_dim = 1

    model = SkipDurationPredictor(input_dim , hidden_dim , num_layers , output_dim)  


    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    num_epochs = 10

    for epoch in range(num_epochs):
        model.train()
        for features, targets in train_dataloader:
            optimizer.zero_grad()
            outputs = model(features)
            loss = criterion(outputs, targets[:, -1].unsqueeze(1))  # Predict only the last target in the sequence
            loss.backward()
            optimizer.step()
        
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {loss.item():.4f}')

    return model

def evaluate(model,test_dataloader):   
    model.eval()
    with torch.no_grad():
        for features, targets in test_dataloader:
            outputs = model(features)
            print(outputs)

def estimator_pipeline(data):
    
    model_features, target, grouped_features, action_encoder, videoID_encoder, topic_encoder, tag_encoder, scaler = pre_process_duration(data)
    sequences = create_sequences(model_features, target, grouped_features)
    train_dataloader,test_dataloader = prepare_data(sequences)
    estimator = prepare_estimator(model_features,train_dataloader)
    evaluate(estimator,test_dataloader)
    return estimator , action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler, data

def estimate_span(estimator ,user_data, action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler, sequence_length=3):
    action_encoded=action_encoder.transform(user_data[['Action']]).toarray()
    topics_encoded = topic_encoder.transform(user_data[['Metadata']].apply(lambda x : x.get('topic'),axis =1).values.reshape(-1,1)).toarray()
    videoID_encoded = videoID_encoder.transform(user_data[['Video_ID']]).toarray()
    tags_combined = user_data['Metadata'].apply(lambda x: ' '.join(x['tags']) if isinstance(x, dict) else x)
    tags_encoded = tag_encoder.transform(tags_combined.values.reshape(-1, 1)).toarray()
    
    action_encoded = action_encoded.reshape(user_data.shape[0], -1)
    videoID_encoded = videoID_encoded.reshape(user_data.shape[0], -1)
    topics_encoded = topics_encoded.reshape(user_data.shape[0], -1)
    tags_encoded = tags_encoded.reshape(user_data.shape[0], -1)
    numerical_features = user_data[['Duration', 'Engagement']].values
    scaled_numerical = scaler.transform(numerical_features)

    model_features = np.hstack((action_encoded, topics_encoded, videoID_encoded, tags_encoded,scaled_numerical ))
    print(model_features)
    

    seq_indices = list(range(len(user_data) - sequence_length, len(user_data)))
    sequence = model_features[seq_indices]
    sequence_tensor = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0) 

    estimator.eval()
    with torch.no_grad():
        output = estimator(sequence_tensor)
        prediction = output.item() 
        
    return prediction

def main():
    model , action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler,df = estimator_pipeline()

    user_id = df['User_ID'].iloc[0]
    user_data = df[df['User_ID'] == user_id].tail(3)  
    
    
    prediction = estimate_span(model ,user_data, action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler)
    
if __name__ =='__main__':
    main()