import pickle as pkl
import json
from random import randint

TOPICS = {'Environment', 'Personal Celebrations & Life Events', 'Politics', 'Religion & Spirituality', 'War and Conflicts', 'Real Estate', 'Shopping', 'Careers', 'Books and Literature', 'Crime, Law & Justice', 'Events & Attractions', 'Communication', 'Sports', 'Video Gaming', 'Disasters', 'Sensitive Topics', 'Food & Drink', 'Education', 'Medical Health', 'Pets', 'Entertainment', 'Travel', 'Automotive', 'Business and Finance', 'Hobbies & Interests', 'Family and Relationships'}

with open('data/programs/programs_info.json', 'r') as f:
    programs = json.load(f)

raw_personas = json.load(open('data/personas/raw_personas.json', 'r'))
personas = []

for p in raw_personas:
    new_p = p.copy()
    new_p["interests"] = {}

    interests = p["interests"]
    watching_time = p["watching_time"]
    
    for topic in interests.keys():
        time = interests[topic]["time"]
        skips = interests[topic]["skips"]

        new_p["interests"][topic] = int(((time / watching_time) * 100)) + int(20/skips if skips != 0 else 0)

    sum_rates = sum([new_p["interests"][topic] for topic in new_p["interests"].keys()])
    
    # ensure that the sum of the rates is 100
    if sum_rates != 100:
        interests_ordered_by_rate = sorted(new_p["interests"].items(), key=lambda x: x[1], reverse=True)
        new_p["interests"][list(interests_ordered_by_rate[0])[0]] += 100 - sum_rates

    personas.append(new_p)

json.dump(personas, open('data/personas.json', 'w'), indent=4)

# transform personas to a list of tuples (persona_name, topic1, rate), (persona_name, topic2, rate), ...
persona_interest_tuples = []

for p in personas:
    for topic in p["interests"].keys():
        persona_interest_tuples.append((p["name"], topic, p["interests"][topic]))

print(persona_interest_tuples)
pkl.dump(persona_interest_tuples, open('data/personas/personas_topics.pkl', 'wb'))