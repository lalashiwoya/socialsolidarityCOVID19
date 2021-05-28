import tweepy
import GetOldTweets3 as got
import datetime
import time
import os
import json
from collections import defaultdict
import glob
import sys

# querys on twitter, can be also placed in a querys.txt
querylist = [
	'#weather',
	'#wetter',
   '#sport',
	'#enconomy',
	'#wirtschaft',
	'#tipp',
	'#tip'
]

# filter for lang, if no filter is needed just provide an empty list
lang = [
    'de',
    'en'
]

# year, month, day
time_since = datetime.datetime(2020, 9, 1)
time_until = datetime.datetime(2020, 9, 15)
time_delta = 1 # the number of days for the next date to query

# GetOldTweets Runs, how often should the GetOldTweets Search be repeated
got_runs = 3

# GetOldTweets Sleep Time, after we get kicked by twitter for too many requests wait for X Minutes
got_sleep_time = 16

# twitter api keys, can also be placed in keyfile.tsv
ACCESS_TOKEN_KEY = ""
ACCESS_TOKEN_SECRET = ""
CONSUMER_KEY = ""
CONSUMER_SECRET = ""

queryfile = "querys.txt"
keyfile = "keyfile.tsv"

# this are vars used by the script

ABSPATH = os.getcwd() + os.path.sep
GOT_FILE_PATH = 'got_id_files' + os.path.sep
TWITTER_TEMP = 'twitter_temp' + os.path.sep
TWITTER_OUTPUT = 'data' + os.path.sep

def get_api_keys():
    keys = {}
    firstLine = True

    with open(ABSPATH + 'keyfile.tsv', 'r') as f:
        lines = f.readlines()

    for line in lines:
        if firstLine:
            firstLine = False
            continue

        parts = line.split('\t')
        
        # exclude no defined keys 
        if parts[1] == 'None':
            continue

        keys[parts[0]] = parts[1].rstrip()

    return keys

def getoldtweets(filename, hashtag):
    days_passed = 0
    days = (time_until-time_since).days
    time_current = time_since
    got_tweets = 0

    while time_current <= time_until:
        try:
            start = time_current.strftime('%Y-%m-%d')
            end = (time_current + datetime.timedelta(days=1)).strftime('%Y-%m-%d')

            print("crawling date", start, "({:2.1%})".format(days_passed / days), end="\r")
        
            tweetCriteria = got.manager.TweetCriteria().setQuerySearch(hashtag).setSince(start).setUntil(end)          
            tweets = got.manager.TweetManager.getTweets(tweetCriteria)

            got_tweets = got_tweets + len(tweets)

            with open(filename, 'a') as f:
                for tweet in tweets:
                    f.write(tweet.id + '\n')
        
            time_current = time_current + datetime.timedelta(days=time_delta)
            days_passed = days_passed + time_delta
        except SystemExit:
            # this block of stdout.write removes the two lines printed by the system.exit
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            sys.stdout.write('\x1b[1A')
            sys.stdout.write('\x1b[2K')
            for i in range(got_sleep_time):
                if i == got_sleep_time - 1:
                    print("Too May Requests: sleeping for", got_sleep_time - i, "minute", end="\r")
                else:
                    print("Too May Requests: sleeping for", got_sleep_time - i, "minutes", end="\r")
                time.sleep(60.0)
                # clear the line to get rid of longer texts
                sys.stdout.write('\x1b[2K')
    
    # clear the line to get rid of longer texts
    print("crawling done... found", got_tweets, "tweets  ")


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def get_datetime_from_string(datetime_string, new_format=""):
    # about formating https://docs.python.org/3/library/datetime.html#strftime-and-strptime-behavior
    dt_obj = datetime.datetime.strptime(datetime_string, '%a %b %d %H:%M:%S %z %Y')

    if new_format == "":
        return dt_obj
    
    return dt_obj.strftime(new_format)

if __name__ == "__main__":
    print("### GetOldTweets Twitter Crawler Version 1.1 ###")

    if os.path.exists(ABSPATH + keyfile):
        print("Load Twitter Api Keys form", keyfile)
        apikeys = get_api_keys()
        CONSUMER_KEY = apikeys['CONSUMER_KEY']
        CONSUMER_SECRET = apikeys['CONSUMER_SECRET']
        ACCESS_TOKEN_KEY = apikeys['ACCESS_TOKEN_KEY']
        ACCESS_TOKEN_SECRET = apikeys['ACCESS_TOKEN_SECRET']

    auth = tweepy.OAuthHandler(CONSUMER_KEY, CONSUMER_SECRET)
    auth.set_access_token(ACCESS_TOKEN_KEY, ACCESS_TOKEN_SECRET)
    api = tweepy.API(auth, wait_on_rate_limit=True, wait_on_rate_limit_notify=True)

    if not os.path.exists(ABSPATH + GOT_FILE_PATH):
        os.mkdir(ABSPATH + GOT_FILE_PATH)
    if not os.path.exists(ABSPATH + TWITTER_TEMP):
        os.mkdir(ABSPATH + TWITTER_TEMP)
    if not os.path.exists(ABSPATH + TWITTER_OUTPUT):
        os.mkdir(ABSPATH + TWITTER_OUTPUT)

    if os.path.exists(ABSPATH + queryfile):
        print("Load querys from", queryfile)
        with open(ABSPATH + queryfile, 'r') as f:
            querylist = [query.rstrip() for query in f.readlines()]

    querys_done = [os.path.splitext(os.path.basename(f))[0] for f in glob.glob(ABSPATH + TWITTER_TEMP + "*.json", recursive=False)]
    if len(querys_done) > 0:
        answer = ""
        while answer not in ['Y', 'N', 'y', 'n', 'Yes', 'No', 'yes', 'no']:
            print("Some of the querys are already done. Continue only with the remaining? (Y/N) ", end="", flush=True)
            answer = input()

        if answer in ['Y', 'y', 'Yes', 'yes']:
            querylist = list(set(querylist).difference(set(querys_done)))
            print("removed", querys_done, "from the querylist")

    print("Start crawling")
    for query in querylist:
        # get tweets for each hashtag from GetOldTweets
        for run in range(got_runs):
            if got_runs > 1:
                print("GoT on query '" + query + "' (" + str(run + 1) + "/" + str(got_runs) + ")")
            else:
                print("GoT on query '" + query + "'")

            filename = ABSPATH + GOT_FILE_PATH + query + "_" + str(run) + ".txt"

            # delete old content of file
            with open(filename, 'w') as f:
                f.write("")

            getoldtweets(filename, "#" + query)

        # get the original data from twitter
        ids = set()
        for run in range(got_runs):
            filename = ABSPATH + GOT_FILE_PATH + query + "_" + str(run) + ".txt"

            with open(filename, 'r') as f:
                lines = [line.rstrip() for line in f.readlines()]

            ids.update(set(lines))

        print(len(ids), "tweets found via GetOldTweets")
        print("getting tweets via Twitter API...", end="\r")

        tweetfile = ABSPATH + TWITTER_TEMP + query + '.json'
        # delete old content of file
        with open(tweetfile, 'w') as f:
            f.write("")

        total = 0

        for i in chunks(list(ids), 100):
            tweets = api.statuses_lookup(i, tweet_mode="extended")
            total += len(tweets)
            with open(tweetfile, 'a') as f:
                for tweet in tweets:
                    # remove non german or english tweets
                    if lang != [] and tweet.lang not in lang:
                        total = total - 1
                        continue

                    j = json.dumps(tweet._json, ensure_ascii=False)
                    f.write(json.dumps(tweet._json, ensure_ascii=False))
                    f.write("\n")

        print(total, "tweets found via the Twitter API after filtering by language")

    print("Crawling Twitter done. Start sorting by month.")
    # order the data by month
    monthfiles = set()
    for query in set(querylist).union(set(querys_done)):
        tweetfile = ABSPATH + TWITTER_TEMP + query + '.json'

        with open(tweetfile, 'r') as f:
            lines = f.readlines()

        for line in lines:
            line_json = json.loads(line)

            month = get_datetime_from_string(line_json['created_at'], "%Y_%m")
            monthfile = ABSPATH + TWITTER_OUTPUT + month + '.json'
            monthfiles.add(monthfile)

            with open(ABSPATH + TWITTER_OUTPUT + month + '.json', 'a') as f:
                f.write(line)

    # sort monthly files by date and filter dublicates
    overall_total_tweets = 0
    for monthfile in monthfiles:
        dublicates = 0
        ids = []

        with open(monthfile, 'r') as f:
            lines = f.readlines()

        sort = defaultdict(int)

        for line in lines:
            line_json = json.loads(line)
    
            if line_json['id'] in ids:
                dublicates = dublicates + 1
                continue
            else:
                ids.append(line_json['id'])

            # we sort by id since this they are ordered by date
            sort[line_json['id']] = line

        print(len(sort), "tweets in", monthfile)
        print(dublicates, "dublicates removed")

        overall_total_tweets = overall_total_tweets + len(sort)

        with open(monthfile, 'w') as f:
            for k, v in sorted(sort.items()):
                f.write(v)

    print("Done... In total", overall_total_tweets, "tweets found for the queries.")