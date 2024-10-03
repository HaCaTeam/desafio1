import json
import pandas as pd
import numpy as np

TOPICS = ['Books and Literature', 'Environment', 'Careers', 'Healthy Living', 'Shopping', 'Automotive', 'Sensitive Topics', 'Politics', 'Events & Attractions', 'Maps & Navigation', 'Personal Celebrations & Life Events', 'Style & Fashion', 'Sports', 'Business and Finance', 'War and Conflicts', 'Hobbies & Interests', 'Education', 'Food & Drink', 'Real Estate', 'Crime, Law & Justice', 'Communication', 'Family and Relationships', 'Disasters', 'Home & Garden', 'Video Gaming', 'Pets', 'Medical Health', 'Entertainment', 'Fine Art', 'Religion & Spirituality', 'Travel']

personas_vectors = []
programs_vectors = []

personas_info = {}
programs_info = {}

with open('data/personas/personas.json', 'rb') as f:
    personas = json.load(f)

    for p in personas:
        persona_vector = [0] * len(TOPICS)
        
        for topic in p["interests"]:
            persona_vector[TOPICS.index(topic)] = p["interests"][topic]
        
        personas_vectors.append({"name": p["name"], "vector": persona_vector})

        personas_info[p["name"]] = p["interests"]
    

with open('data/programs/programs_info.json', 'rb') as f:
    programs = json.load(f)

    for p in programs:
        program_vector = [0] * len(TOPICS)
        
        for topic in p["Topics"]:
            program_vector[TOPICS.index(topic["description"])] = topic["percentage"]

        programs_vectors.append({"name": p["Title"], "vector": program_vector})

        programs_info[p["Title"]] = p

# compute the similarity between each persona and each program
similarity_matrix = []

for p in personas_vectors:
    row = {}

    for pr in programs_vectors:
        similarity = np.dot(p["vector"], pr["vector"]) / (np.linalg.norm(p["vector"]) * np.linalg.norm(pr["vector"])) # cosine similarity

        row[pr["name"]] = similarity

    similarity_matrix.append(row)

df = pd.DataFrame(similarity_matrix, index=[p["name"] for p in personas_vectors])

def sort_programs_by_similarity(persona_name):
    return df.loc[persona_name].sort_values(ascending=False)

# export for "David"
programs = sort_programs_by_similarity("David").head(10)

programs_obj = [programs_info[program] for program in programs.index]

data_obj = {
    "preferences": personas_info["David"],
    "programs": programs_obj
}

with open('data/david_example.json', 'w') as f:
    json.dump(data_obj, f, indent=4)

