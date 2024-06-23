import os
import pandas as pd

# if is csv files
files = [f for f in os.listdir('./dataset/reddit-dataset') if f.endswith('.csv')]

initial_path = './dataset/reddit-dataset/'
save_path = './dataset/formatted-reddit-dataset/'

headers = ["text","id","subreddit","meta","time","author","ups","downs","authorlinkkarma","authorkarma","authorisgold"]

for file in files:
    print(f"{initial_path}{file}")
    df = pd.read_csv(f"{initial_path}{file}")
    # drop unnamed columns
    df = df.loc[:, ~df.columns.str.contains('^Unnamed')]
    print(df.columns)
    if len(df.columns) != len(headers):
        #drop first column
        df = df.drop(df.columns[0], axis=1)
    
    #rename headers
    df.columns = headers
    # drop first row
    df = df.iloc[1:]
    
    #save to new file
    df.to_csv(f"{save_path}{file}", index=False)


    print(df.head())
