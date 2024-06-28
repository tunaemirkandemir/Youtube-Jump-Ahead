import random
import string
from datetime import datetime, timedelta

def gen_datetime(min_year, max_year=datetime.now().year):
    # generate a datetime in format yyyy-mm-dd hh:mm:ss.000000
    start = datetime(min_year, 1, 1, 00, 00, 00)
    years = max_year - min_year + 1
    end = start + timedelta(days=365 * years)
    return start + (end - start) * random.random()


def create_data():
    userID = random.randint(10,70)
    videoID = random.randint(1,5)
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

def create_dummy_data(amount=400):
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
        
    
    
    return data 





