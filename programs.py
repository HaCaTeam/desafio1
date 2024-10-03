import pickle as pkl
import requests
import json

STATION_ID = 327250 # RTP1

# load thematics from json file
with open('data/contents.json', 'r') as f:
    data = json.load(f)["value"]

    programs = [
        {
            "Title": p["Title"],
            "Synopsis": p["Synopsis"],
            "StartDate": p["StartDate"],
            "EndDate": p["EndDate"],
        }
        for p in data
    ]

# get topics vector for each program
API_KEY = "dGVhbWQ6YmViYmIyNDg1MTFkNDlkMzgzMDE4YmYwYmFiZWFmZjQ="

def get_topic_segments(station_id, start_time, end_time):
    url = f"https://mediadive.poc.alticelabs.com/pubblocks/topics/?stationId={station_id}&startTime={start_time}&endTime={end_time}"
    header = {
        "Authorization": f"Basic {API_KEY}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36"
    }

    response = requests.get(url, headers=header)
    return response.json()["topicPercentage"]


set_topics = set()
topics = []

for program in programs:
    start_time, end_time = program["StartDate"], program["EndDate"]
    
    res = get_topic_segments(STATION_ID, start_time, end_time)
    program = {**program, "Topics": res}
    
    topics.append(program)

    set_topics = set_topics.union(set([t["description"] for t in res]))

print(set_topics)

json.dump(topics, open('data/programs/programs_info.json', 'w'), indent=4)

# transform programs to a list of tuples (program_name, topic1, rate), (program_name, topic2, rate), ...
program_interest_tuples = []

for p in topics:
    for topic in p["Topics"]:
        program_interest_tuples.append((p["Title"], topic["description"], topic["percentage"]))

pkl.dump(program_interest_tuples, open('data/programs/program_topics.pkl', 'wb'))