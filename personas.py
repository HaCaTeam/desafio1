import json
from random import randint

TOPICS = {'Books and Literature', 'Environment', 'Careers', 'Healthy Living', 'Shopping', 'Automotive', 'Sensitive Topics', 'Politics', 'Events & Attractions', 'Maps & Navigation', 'Personal Celebrations & Life Events', 'Style & Fashion', 'Sports', 'Business and Finance', 'War and Conflicts', 'Hobbies & Interests', 'Education', 'Food & Drink', 'Real Estate', 'Crime, Law & Justice', 'Communication', 'Family and Relationships', 'Disasters', 'Home & Garden', 'Video Gaming', 'Pets', 'Medical Health', 'Entertainment', 'Fine Art', 'Religion & Spirituality', 'Travel'}

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
