import os
import json
import csv
import random
import requests
import re
from LLM import LLM
from dotenv import load_dotenv
load_dotenv()


BASE_URL = "https://mediadive.poc.alticelabs.com/pubblocks"
API_KEY = os.getenv("API_KEY")
STATION_ID = 327250

PORT = 3002


def update_html_file(new_image_url, new_box_text):
    file_path = 'conteudos_rapidos/image.html'
    with open(file_path, 'r', encoding='utf-8') as file:
        content = file.read()

    content = re.sub(r'const imageUrl = ".*?";', f'const imageUrl = "{new_image_url}";', content)
    content = re.sub(r'const boxText = ".*?";', f'const boxText = "{new_box_text}";', content)

    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)

    print(f'http://localhost:{PORT}/' + file_path)


def clip_transcript(transcripts, clip):
    start_period = clip['start_period'].replace("T", " ").replace("Z", "")
    end_period = clip['end_period'].replace("T", " ").replace("Z", "")

    clip_transcripts = []
    start = False
    for row in transcripts:
        if not start and row[0].split(".")[0] == start_period:
            start = True
            clip_transcripts.append(row[2])
        elif start and row[1].split(".")[0] == end_period:
            clip_transcripts.append(row[2])
            break
        elif start:
            clip_transcripts.append(row[2])

    return clip_transcripts


def clip_thumbnail(clip):
    start_period = clip['start_period']
    end_period = clip['end_period']
    url = f"{BASE_URL}/thumbnails?stationId={STATION_ID}&startTime={start_period}&endTime={end_period}&limit=5"
    header = {
        "Authorization": f"Basic {API_KEY}",
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/104.0.5112.79 Safari/537.36"
    }
    response = requests.get(url, headers=header)
    response = response.json()

    return random.choice(response['thumbnails'])['url'] if response['thumbnails'] else None


if __name__ == "__main__":

    with open('data/clips.json') as f:
        clips = json.load(f)

    with open('data/transcripts.csv') as f:
        content = csv.reader(f, delimiter=';')
        next(content)
        transcripts = [(row[1], row[2], row[-1]) for row in content]

    transcripts = [clip_transcript(transcripts, clip) for clip in clips]

    llm = LLM()

    for i, t in enumerate(transcripts[:5]):
        title, summary, keywords = llm.process_transcripts(t)   # only one clip for testing
        print(f"\nTitle: \n{title}")
        print(f"\nSummary: \n{summary}")
        print(f"\nKeywords: \n{keywords}")

        thumbnail = clip_thumbnail(clips[i])
        print(f"\nThumbnail: \n{thumbnail}")
        update_html_file(thumbnail, title[1:-1])

        print("\n\n")
        print("-" * 100)
        print("\n\n")