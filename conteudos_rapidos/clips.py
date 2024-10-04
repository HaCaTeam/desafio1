import os
import json
import requests
from dotenv import load_dotenv
load_dotenv()


BASE_URL = "https://mediadive.poc.alticelabs.com/pubblocks"
API_KEY = os.getenv("API_KEY")
STATION_ID = 327250


def clips_details(station_id, start_time, end_time):
    url = f"{BASE_URL}/topics?stationId={station_id}&startTime={start_time}&endTime={end_time}"
    header = {
        "Authorization": f"Basic {API_KEY}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36"
    }
    response = requests.get(url, headers=header)
    response = response.json()

    return response['topicSegmentGroup']


if __name__ == "__main__":

    # load profile
    with open('data/david_example.json') as f:
        profile = json.load(f)

    preferences = profile['preferences']
    programs = profile['programs']

    # get clips of each program
    clips = []
    for program in programs:
        startTime = program['StartDate']
        endTime = program['EndDate']

        details = clips_details(STATION_ID, startTime, endTime)

        for clip in details:
            # ignore very short clips
            if clip['n_segments'] < 2:
                continue

            description = clip['description']
            if description in preferences:
                clip['rank'] = preferences[description]
            else:
                clip['rank'] = 0

            clips.append(clip)

    # sort clips by rank
    clips.sort(key=lambda x: x['rank'], reverse=True)

    # save clips as json file
    with open("clips.json", 'w') as f:
        json.dump(clips, f, indent=4)
