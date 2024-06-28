import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder,StandardScaler
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
from dummy_data_generator import create_dummy_data

def get_data(num_rows=1000):
    data = create_dummy_data(num_rows)
    
    df = pd.DataFrame(data)
    print(df[df['User_ID']== 10].head(10))
    return df

def pre_process_df(df , fit=True):
    df.sort_values(by=['User_ID', 'Timestamp'], inplace=True)

    df['Skipped'] = (df['Duration_Skipped'] > 0).astype(int)

     # Initialize encoders
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
    target = df['Skipped'].values
    grouped_features = list(df.groupby('User_ID').indices.values())

    return model_features, target, grouped_features, action_encoder, videoID_encoder, topic_encoder, tag_encoder, scaler


def create_sequences(model_features, target, grouped_features,sequence_length =3):
    sequences =[]

    for indices in grouped_features:

        if len(indices) < sequence_length:
            continue

        for i in range(len(indices) - sequence_length + 1):
            seq_indices = indices[i:i + sequence_length]
            sequences.append((model_features[seq_indices], target[seq_indices]))
    
    return sequences
class SequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __getitem__(self, idx):
        features, targets = self.sequences[idx]
        return torch.tensor(features, dtype=torch.float32), torch.tensor(targets, dtype=torch.float32)


    def __len__(self):
        return len(self.sequences)


def prepare_data(sequences):
    Train_sequences, Test_sequences = train_test_split(sequences, test_size=0.2, random_state=42,shuffle=False)
        
    train_dataset = SequenceDataset(Train_sequences)
    test_dataset = SequenceDataset(Test_sequences)

    train_dataloader = DataLoader(train_dataset, batch_size=32,shuffle= True)
    test_dataloader = DataLoader(test_dataset, batch_size=32,shuffle = False)
    return train_dataloader,test_dataloader

class JumpRegressorRNN(nn.Module):
    def __init__(self , input_dim , hidden_dim , num_layers , output_dim):
        super(JumpRegressorRNN , self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.rnn = nn.RNN(input_dim,hidden_dim , num_layers , batch_first = True)
        self.fc = nn.Linear(hidden_dim , output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self , x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim).to(x.device)
        out, _ = self.rnn(x, h0)
        out = self.fc(out[:, -1, :])
        out = self.sigmoid(out)
        return out
    
def prepare_model(model_features,train_dataloader):

    input_dim = model_features.shape[1]
    hidden_dim = 64
    num_layers = 2
    output_dim = 1

    model = JumpRegressorRNN(input_dim , hidden_dim , num_layers , output_dim)  


    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    epochs = 10
    for epoch in range(epochs):
        model.train()
        for features, target in train_dataloader:
            optimizer.zero_grad()
            output = model(features)
            loss = criterion(output, target[:, -1].unsqueeze(1))
            loss.backward()
            optimizer.step()

        print(f"Epoch {epoch+1}, Loss: {loss.item()}")

    return model




#evaluate OK
#create a pipeline
def evaluate(model,test_dataloader):
    model.eval()
    with torch.no_grad():
        correct = 0
        total = 0
        for features, targets in test_dataloader:
            outputs = model(features)
            predicted = (outputs > 0.5).float()
            total += targets.size(0)
            correct += (predicted[:, -1] == targets[:, -1]).sum().item()

        accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')


def regressor_pipeline(df):
    
    model_features, target, grouped_features, action_encoder, videoID_encoder, topic_encoder, tag_encoder, scaler= pre_process_df(df)
    sequences = create_sequences(model_features, target, grouped_features)
    train_dataloader,test_dataloader = prepare_data(sequences)
    model = prepare_model(model_features,train_dataloader)
    evaluate(model,test_dataloader) 
    return model , action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler,df
    
def predict_skip(model ,user_data, action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler, sequence_length=3):
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

    model.eval()
    with torch.no_grad():
        output = model(sequence_tensor)
        prediction = (output > 0.5).float().item()  
        
    return prediction

def main():
    model , action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler,df = regressor_pipeline()

    user_id = df['User_ID'].iloc[0]
    user_data = df[df['User_ID'] == user_id].tail(3)  
    
    print(user_data)
    print(user_id)
    prediction = predict_skip(model ,user_data, action_encoder, videoID_encoder, topic_encoder, tag_encoder,scaler)
    print(prediction)
    

if __name__ == "__main__":
    main()
    


