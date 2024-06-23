import os
import pandas as pd


initial_path = './dataset/reddit-dataset/'
save_path = './dataset/formatted-reddit-dataset/'


files = [f for f in os.listdir(initial_path) if f.endswith('.csv')]

os.mkdir(save_path)


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
    # drop all rows with just white space in text
    df = df[df['text'].str.strip().astype(bool)]
    df = df.dropna()

    df.drop(columns=['id','author',], inplace=True)
    # only first 1000 rows
    df = df.head(100)
    
    #save to new file
    df.to_csv(f"{save_path}{file}", index=False)


    print(df.head())

