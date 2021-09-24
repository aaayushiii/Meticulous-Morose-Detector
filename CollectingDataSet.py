import pandas as pd
import twint
import pandas as pd
import re
from google.colab import drive
import matplotlib as plt
import seaborn as sns
drive.mount('/content/gdrive')
depress_tags = ["#depressed", "#depression", "#loneliness", "#hopelessness"]

# Collecting tweets from the year 2015
content = {}
for i in range(len(depress_tags)):
    print(depress_tags[i])
    c = twint.Config()
    c.Format = "Tweet id: {id} | Tweet: {tweet}"
    c.Search = depress_tags[i]
    c.Limit = 1000
    c.Year = 2015
    c.Store_csv = True
    c.Store_Object = True
    c.Output = "/content/gdrive/My Drive/data/dataset_en_all7.csv"
    c.Hide_output = True
    c.Stats = True
    c.Lowercase  = True
    c.Filter_retweets = True
    twint.run.Search(c)

depress_tags = ["#depressed", "#depression"]

# Collecting tweets from the year 2016
content = {}
for i in range(len(depress_tags)):
    c = twint.Config()
    c.Format = "Tweet id: {id} | Tweet: {tweet}"
    c.Search = depress_tags[i]
    c.Limit = 1000
    c.Year = 2016
    c.Store_csv = True
    c.Store_Object = True
    c.Output = "/content/gdrive/My Drive/data/dataset_en_al19.csv"
    c.Hide_output = True
    c.Stats = True
    c.Lowercase  = True
    twint.run.Search(c)

# Reading the collecting data in the form of data frames
df1 = pd.read_csv("/content/gdrive/My Drive/data/dataset_en_all7.csv")
df2 = pd.read_csv("/content/gdrive/My Drive/data/dataset_en_al19.csv")
df_all = pd.concat([df1, df2])
len(df1), len(df2), len(df_all)

# Combine dataset and remove duplicates based on id and tweet content
df_all = df_all.drop_duplicates(subset =["id"])
df_all.shape
pd.set_option('display.max_colwidth', -1)
df_all.head()

# Filtering the dataset
selection_to_remove = ["#mentalhealth", "#health", "#happiness", "#mentalillness", "#happy", "#joy", "#wellbeing"]

# remove entries that contain positive, or medical sounding tags
mask1 = df_all.hashtags.apply(lambda x: any(item for item in selection_to_remove if item in x))
df_all[mask1].tweet.tail()
df_all = df_all[mask1==False]

# remove entries with more than three hashtags, as it may be promotional messages
mask2 = df_all.hashtags.apply(lambda x: x.count("#") < 4)
df_all = df_all[mask2]

# remove tweets with at mentions as they are sometimes retweets
mask3 = df_all.mentions.apply(lambda x: len(x) < 5)
df_all = df_all[mask3]

# remove entries with less than x chars / words
mask4a = df_all.tweet.apply(lambda x: len(x) > 25)
df_all = df_all[mask4a]
mask4b = df_all.tweet.apply(lambda x: x.count(" ") > 5)
df_all = df_all[mask4b]

# remove entries containing urls - as they are likely to be promotional messages
mask5 = df_all.urls.apply(lambda x: len(x) < 5)
df_all = df_all[mask5]

# Removing hashtags
df_all["mod_text"] = df_all["tweet"].apply(lambda x: re.sub(r'#\w+', '', x))
col_list = ["id", "conversation_id", "date", "username", "mod_text", "hashtags", "tweet"]
df_final1 = df_all[col_list]
df_final1 = df_final1.rename(columns={"mod_text": "tweet_processed", "tweet": "tweet_original"})
df_final1["target"] = 1

# Segregating into 3 csv files
df_final1_1 = df_final1[:400]
df_final1_2 = df_final1[400:800]
df_final1_3 = df_final1[800:]
df_final1.to_csv("/content/gdrive/My Drive/data/tweets_final.csv")

# Saving to csv
df_final1_1.to_csv("/content/gdrive/My Drive/data/tweets_final_1.csv")
df_final1_2.to_csv("/content/gdrive/My Drive/data/tweets_final_2.csv")
df_final1_3.to_csv("/content/gdrive/My Drive/data/tweets_final_3.csv")

df_all.to_csv("/content/gdrive/My Drive/data/tweets_v3.csv")
users = df_all.username

# content = {}
# for i in users:
#     c = twint.Config()
#     c.Search = "#depressed"
#     c.Username = "noneprivacy"
#     c.Username = i
#     c.Format = "Tweet id: {id} | Tweet: {tweet}"
#     c.Limit = 100
#     c.Store_csv = True
#     c.Store_Object = True
#     c.Output = "/content/gdrive/My Drive/data/dataset_v3.csv"
#     c.Hide_output = True
#     c.Stats = True
#     c.Lowercase  = True
#     twint.run.Search(c)

df_anger = pd.read_csv("2018-EI-reg-En-anger-test-gold.txt", delimiter="\t")
df_fear = pd.read_csv("2018-EI-reg-En-fear-test-gold.txt", delimiter="\t")
df_joy = pd.read_csv("2018-EI-reg-En-joy-test-gold.txt", delimiter="\t")
df_sadness = pd.read_csv("2018-EI-reg-En-sadness-test-gold.txt", delimiter="\t")

pd.set_option('display.max_colwidth', -1)

# Non-depressive tweets
df_non_depress = pd.concat([df_anger1[df_anger1["Intensity Score"] < 0.3], df_fear1[df_fear1["Intensity Score"] < 0.4], df_joy1[df_joy1["Intensity Score"] > 0.5], df_sadness1[df_sadness1["Intensity Score"] < 0.3]])

# More depressive tweets
df_depress = df_sadness1[df_sadness1["Intensity Score"] > 0.8]
len(df_depress)

df_non_depress["target"] = 0
df_depress["target"] = 1

# Concatinating the final dataset
df_final = pd.concat([df_non_depress, df_depress])
df_final = df_final.sample(frac=1).reset_index(drop=True)

# Creating the new dataset
df_final.to_csv("general_tweets.csv")

# Read operation
df0 = pd.read_csv("./data/general_tweets.csv")
df1 = pd.read_csv("./data/tweets_final_1_clean.csv", engine='python')
df2 = pd.read_csv("./data/tweets_final_2_clean.csv", engine='python')
df3 = pd.read_csv("./data/tweets_final_3_clean.csv", engine='python')
df4 = pd.read_csv("./data/tweets_final_4_clean.csv", engine='python')
df5 = pd.read_csv("./data/tweets_final_5_clean.csv", engine='python')
df6 = pd.read_csv("./data/tweets_final_6_clean.csv", engine='python')
pd.set_option('display.max_colwidth', -1)

df0 = df0[['Tweet','target']].copy()

df1 = df1[['tweet_processed','target']].copy()
df2 = df2[['tweet_processed','target']].copy()
df3 = df3[['tweet_processed','target']].copy()
df4 = df4[['tweet_processed','target']].copy()
df5 = df5[['tweet_processed','target']].copy()
df6 = df6[['tweet_processed','target']].copy()

df0 = df0.rename(columns = {"Tweet": "tweet"})

df1 = df1.rename(columns = {"tweet_processed": "tweet"})
df2 = df2.rename(columns = {"tweet_processed": "tweet"})
df3 = df3.rename(columns = {"tweet_processed": "tweet"})
df4 = df4.rename(columns = {"tweet_processed": "tweet"})
df5 = df5.rename(columns = {"tweet_processed": "tweet"})
df6 = df6.rename(columns = {"tweet_processed": "tweet"})

df_all = pd.concat([df0, df1, df2, df3, df4, df5, df6])

df_all = df_all.sample(frac=1).reset_index(drop=True)

# Save the final dataset containing positive tweets and negative tweets
df_all.to_csv("./data/tweets_combined.csv")
