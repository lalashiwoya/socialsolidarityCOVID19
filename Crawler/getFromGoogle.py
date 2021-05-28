import requests
import json
import os
import glob

TWITTER_DATA = 'data'
OUTPUT_FILE = 'twitter_locations_from_google.json'

ABSPATH = os.getcwd() + os.path.sep

def getApiKeyFromFile(keyfile="apikey.txt"):
    """
    Reads the GoogleApi key from a file. The file should just contain the api key and nothing else.

    Args:
        keyfile (str, optional): Path to the file containing the GoogleApi key. Defaults to "apikey.txt".

    Returns:
        str: The GoogleApi key from the given file.
    """

    with open(ABSPATH + keyfile, 'r') as f:
        return f.readline()

def getFromGoogle(keyword, api_key, fields="formatted_address,name,geometry"):
    """
    Queries the Google Places Api for a given location.

    Args:
        keyword (str): The location the api should be queried for.
        api_key (str): A GoogleApi key with the rights to use the GoogleP Places Api.
        fields (str, optional): The fields the api should return. See https://developers.google.com/places/web-service/place-data-fields#places-api-fields-support for more information. Defaults to "formatted_address,name,geometry".

    Returns:
        dict: The location data returned by the Google api.
    """

    url = "https://maps.googleapis.com/maps/api/place/findplacefromtext/json?input=" + keyword + "&inputtype=textquery&fields=" + fields + "&key=" + api_key
    return requests.get(url).json()

def stripDataFromGoogle(data):
    """
    Extracts the geo coordinates from the json returned by the Google api. Also checks if the json is valid. If the json contains more than on set of coordinates no coordinate will be returned.

    Args:
        data (dict): The json data provided by the Google Places Api.

    Returns:
        [dict]: A dict with geo coodinates in longitude and latitude. Or "False" if the json was not valid or there are multible coordinates in the json.
    """

    if data['status'] == "OK" and 'candidates' in data and len(data['candidates']) == 1:
        return data['candidates'][0]

    return False

def loadDictFromJson(dictfile):
    """
    Loads previously queried locations for the json file.

    Args:
        dictfile (str): The path to the locations file.

    Returns:
        dict: The previously queried locations that are saved in the given file.
    """

    if not os.path.exists(ABSPATH + dictfile):
        return {}

    with open(ABSPATH + dictfile, 'r') as f:
        return json.loads(f.read())

def saveDictToJson(dictfile, to_json):
    """
    Saves the given dict at given location. If the file already exists it will be overwritten.

    Args:
        dictfile (str): The path to the location where the data should be saved.
        to_json (dict): The data that should be saved.
    """

    with open(ABSPATH + dictfile, 'w') as f:
        f.write(json.dumps(to_json, ensure_ascii=False))

def getTwitterFilesFromFolder(folder="../TwitterData"):
    """
    Searches for .json files at the given location.

    Args:
        folder (str, optional): The location that should be searched. Defaults to "../TwitterData".

    Returns:
        list(str): A list of pathes to .json files.
    """
    return [folder + os.path.sep + os.path.splitext(os.path.basename(f))[0] + ".json" for f in glob.glob(folder + "/*.json", recursive=False)]

if __name__ == "__main__":
    KEY = getApiKeyFromFile()
    dictfile = OUTPUT_FILE

    twitter_data_files = getTwitterFilesFromFolder(TWITTER_DATA)
    location_dict = loadDictFromJson(dictfile)

    for twitter_data_file in twitter_data_files:
        print('find locations for', twitter_data_file, '...')
        cur_loc = len(location_dict)

        with open(twitter_data_file, 'r') as f:
            for json_line in f.readlines():
                twitter_data = json.loads(json_line)

                location = twitter_data['user']['location']
                if location.strip() == '' or location in location_dict:
                    continue

                google_location = getFromGoogle(location, KEY)
                location_dict[location] = stripDataFromGoogle(google_location)

        print(len(location_dict) - cur_loc, 'new locations found')

        saveDictToJson(dictfile, location_dict)