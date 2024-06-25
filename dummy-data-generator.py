import random
import string
from datetime import datetime, timedelta
import csv
import os

def file_is_empty(path):
    return os.stat(path).st_size==0



def gen_datetime(min_year, max_year=datetime.now().year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()
'''
data = {
    'User_ID': [12345, 12345, 12345, 67890, 67890, 67890],
    'Video_ID': ['abc123', 'abc123', 'abc123', 'def456', 'def456', 'def456'],
    'Action': ['play', 'skip', 'pause', 'play', 'skip', 'pause'],
    'Timestamp': ['2024-06-01 10:00:00', '2024-06-01 10:05:00', '2024-06-01 10:10:00', '2024-06-01 10:15:00', '2024-06-01 10:20:00', '2024-06-01 10:25:00'],
    'Segment_Start': ['00:00:00', '00:05:00', '00:10:00', '00:00:00', '00:05:00', '00:10:00'],
    'Segment_End': ['00:05:00', '00:07:00', '00:12:00', '00:05:00', '00:07:00', '00:12:00'],
    'Duration': [300, 120, 120, 300, 120, 120],
    'Engagement': [5, 3, 4, 5, 3, 4],
    'Feedback': ['positive', 'neutral', 'positive', 'positive', 'neutral', 'positive'],
    'Metadata': [{'topic': 'tech', 'tags': ['AI', 'ML']}, {'topic': 'tech', 'tags': ['AI', 'ML']}, {'topic': 'tech', 'tags': ['AI', 'ML']},
                 {'topic': 'music', 'tags': ['jazz', 'blues']}, {'topic': 'music', 'tags': ['jazz', 'blues']}, {'topic': 'music', 'tags': ['jazz', 'blues']}],
    'Duration_Skipped': [0, 120, 60, 0, 120, 60]  
} '''

def create_data():
    userID = random.randint(10000,99999)
    videoID = random.choice(string.ascii_letters) + random.choice(string.ascii_letters) +random.choice(string.ascii_letters)+ str(random.randint(100,999))
    action = random.choice(['play', 'skip', 'pause'])
    timestamp = gen_datetime(2020).strftime("%Y-%m-%d %H:%M:%S")
    segment_start = "00:"+gen_datetime(2020).strftime("%M:%S")
    duration = random.randint(1,600)

    saat = duration / (60**2)
    dak = (duration % (60**2)) / 60
    sn = (duration % (60**2)) % 60
    segment_end = str(int(segment_start[:2])+int(saat) % 100).zfill(2) +":"+str(int(segment_start[3:5])+int(dak) % 100).zfill(2)+":"+str(int(segment_start[6:8])+int(sn) % 100).zfill(2)

    if action == 'play':
        engagement = 5
        feedback = 'positive'
        duration_skipped = 0
    elif action =='skip':
        engagement = 3
        feedback = 'neutral'
        duration_skipped = duration
    else :
        duration_skipped = 0
        engagement = 4
        feedback = 'positive'


    topic = random.choice(['tech', 'music','politics','food'])
    tags_arr = random.sample(['AI', 'ML', 'data science', 'deep learning','jazz','rock','blues','pizza','pasta','burger'], 3)

    metadata = str({'topic': topic, 'tags': tags_arr})

    return [userID, videoID, action, timestamp, segment_start, segment_end, duration, engagement, feedback, metadata, duration_skipped]



data = {
    'User_ID': [],
    'Video_ID': [],
    'Action': [],
    'Timestamp': [],
    'Segment_Start': [],
    'Segment_End': [],
    'Duration': [],
    'Engagement': [],
    'Feedback': [],
    'Metadata': [],
    'Duration_Skipped': []  }



def create_dummy_data(amount=400):
    for epoch in range(amount):
        result = create_data()
        data['User_ID'].append(result[0])
        data['Video_ID'].append(result[1])
        data['Action'].append(result[2])
        data['Timestamp'].append(result[3])
        data['Segment_Start'].append(result[4])
        data['Segment_End'].append(result[5])
        data['Duration'].append(result[6])
        data['Engagement'].append(result[7])
        data['Feedback'].append(result[8])
        data['Metadata'].append(result[9])
        data['Duration_Skipped'].append(result[10])
        print(data.values)
    
    keys = data.keys()
    values = list(zip(*data.values()))
    if file_is_empty('dummy_data.csv'):
        with open('dummy_data.csv', 'w',newline='') as output_file:
            dict_writer = csv.writer(output_file)
            dict_writer.writerow(keys)
            dict_writer.writerows(values)
    else:
        with open('dummy_data.csv', 'a',newline='') as output_file:
            dict_writer = csv.writer(output_file)   
            dict_writer.writerows(values)
        
   

        
    


def main():
    create_dummy_data(10)


if __name__ == '__main__':
    main()




