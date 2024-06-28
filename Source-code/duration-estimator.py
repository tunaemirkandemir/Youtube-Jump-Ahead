
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
import torch

import torch.nn as nn
import torch.optim as optim
from jump_regressor import  get_data,create_sequences,prepare_data,evaluate

def pre_process_duration(df):

    df.sort_values(by=['User_ID', 'Timestamp'], inplace=True)

    encoder = OneHotEncoder()
    action_encoded=encoder.fit_transform(df[['Action']]).toarray()
    topics_encoded = encoder.fit_transform(df[['Metadata']].apply(lambda x : x.get('topic'),axis =1).values.reshape(-1,1)).toarray()
    tags_combined = df['Metadata'].apply(lambda x: ' '.join(x['tags']) if isinstance(x, dict) else x)
    tags_encoded = encoder.fit_transform(tags_combined.values.reshape(-1, 1)).toarray()

    model_features = np.hstack((action_encoded,topics_encoded,tags_encoded,df[['Duration','Engagement']].values))

    target = df['Duration_Skipped'].values

    grouped_features = list(df.groupby('User_ID').indices.values())
    

    return model_features, target, grouped_features

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


data=get_data()
model_features, target, grouped_features = pre_process_duration(data)
sequences = create_sequences(model_features, target, grouped_features)
train_dataloader,test_dataloader = prepare_data(sequences)
estimator = prepare_estimator(model_features,train_dataloader)

