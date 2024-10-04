import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()


BASE_URL = "http://ott.online.meo.pt/program/v9/Programs/DvrLiveChannelPrograms?$format=json&UserAgent=AND_NG&$filter=StartDate ge datetime'2024-09-29T00:00:00' and StartDate lt datetime'2024-10-03T22:59:00' and CallLetter eq 'RTP1' and IsEnabled eq true and IsLiveAnytimeChannel eq true and IsAdultContent eq false and IsBlackout eq false and substringof('Mobile',AvailableOnChannels)&$orderby=StartDate desc&$inlinecount=allpages"

response = requests.get(BASE_URL)
response = response.json()

all_programs = response['value']

while response.get('odata.nextLink'):
    response = requests.get(response['odata.nextLink'])
    response = response.json()
    all_programs.extend(response['value'])

programs = {
    "value": all_programs
}

with open('data/contents.json', 'w') as f:
    json.dump(programs, f, indent=4)
