"""
module to process raw LTL to canonical LTL 
"""

import json

def read_lines_from_file(filename):
    """Read lines from a file and return them as a list."""
    with open(filename, 'r') as file:
        return file.read().splitlines()

def print_unique_ltls(unique_ltls):
    """Print sorted unique LTL expressions based on their length."""
    for ltl in sorted(unique_ltls, key=lambda x: len(x)):
        print(ltl)

def print_data_point_info(data_points, unique_ltls):
    """Print information about the data points."""
    print("Clean-up Domain, with augmentation")
    print("Number of Data Points", len(data_points))
    print("Number of unique LTL expressions:", len(unique_ltls))
    print("Number of unique LTL structures:", 4)

def print_unique_lengths(data_points):
    """Print unique LTL expressions based on their length."""
    seen_length = set()
    for dp in data_points:
        if len(dp['ltl']) not in seen_length:
            seen_length.add(len(dp['ltl']))
            print(dp['ltl'])
            print(dp['eng'])
            print()

def build_translation_seeds(atomic_props):
    """Build and return a list of translation seeds based on atomic propositions."""
    translation_seeds = []
    for r1 in atomic_props:
        translation_seeds.append(build_type_a(r1))
        for r2 in atomic_props:
            if r1 == r2:
                continue
            translation_seeds.append(build_type_b(r1, r2))
            translation_seeds.append(build_type_c(r1, r2))
            for r3 in atomic_props:
                if r1 == r3 or r2 == r3:
                    continue
                translation_seeds.append(build_type_d(r1, r2, r3))
    return translation_seeds

def build_type_A(ap1):
    return {
        'raw_ltl': f"F {ap1['ap']}",
        'canonical_ltl': f"finally {ap1['eng']}",
        'eng': f"{ap1['eng']}"}

def build_type_B(room_1, room_2):
    return {
        "raw_ltl": f"F & {room_1['ap']} F {room_2['ap']}",
        "canonical_ltl": f"finally ( and (  {room_1['eng']} , finally ( {room_2['eng']} )  )  )",
        "eng": f"{room_1['eng']} first, and then {room_2['eng']}"}

def build_type_C(room_1, room_2):
    return {
        "raw_ltl": f"& F {room_1['ap']} G ! {room_2['ap']}",
        "canonical_ltl": f"and ( finally ( {room_1['eng']} ) , globally ( not ( {room_2['eng']} ) ) )",
        "eng": f"{room_1['eng']}, and do not ever {room_2['eng']}"
    }

def build_type_D(room_1, room_2, room_3):
    return {
        "raw_ltl": f"F & | {room_1['ap']} {room_2['ap']} F {room_3['ap']}",
        "canonical_ltl": f"finally ( and ( or ( {room_1['eng']} , {room_2['eng']} ) , finally ( {room_3['eng']} ) ) )",
        "eng": f"{room_1['eng']} or {room_2['eng']} to finally {room_3['eng']}"
    }

def write_json(data, filename):
    """Write data to a JSON file."""
    with open(filename, 'w') as file:
        json.dump(data, file, indent=2)

def process_data_points(data_points, translation_seeds):
    """Process data points and write them to a file."""
    raw_canonical_mapping = {
        seed['raw_ltl']: seed['canonical_ltl'] for seed in translation_seeds
    }
    with open("output.jsonl", "w") as file:
        for dp in data_points:
            entry = {
                'canonical': raw_canonical_mapping[dp['ltl']],
                'formula': raw_canonical_mapping[dp['ltl']],
                'natural': dp['eng'],
                'raw_ltl': dp['ltl'],
            }
            json.dump(entry, file)
            file.write('\n')

# Main execution logic
raw_ltls = read_lines_from_file("raw.txt")
unique_ltls = set(raw_ltls)
raw_engs = read_lines_from_file("nl.txt")

data_points = [{'ltl': ltl, 'eng': eng} for ltl, eng in zip(raw_ltls, raw_engs)]

print_unique_ltls(unique_ltls)
print_data_point_info(data_points, unique_ltls)
print_unique_lengths(data_points)

room_types = [
    {'ap': 'B', 'eng': 'go to the blue room'},
    {'ap': 'R', 'eng': 'go to the red room'},
    {'ap': 'Y', 'eng': 'go to the yellow room'},
    {'ap': 'C', 'eng': 'go to the green room'},
]

translation_seeds = build_translation_seeds(room_types)

print(len(translation_seeds))

seed_ltls = set([t['raw_ltl'] for t in translation_seeds])
print(unique_ltls - seed_ltls)

additional_seed_1 = {
    "raw_ltl": "F & R F X",
    "canonical_ltl": "finally ( and (  go to the red room , finally ( go to the blue room with chair )  )  )",
    "eng": "go to the red room and push the chair into the blue room"
}

additional_seed_2 = {
    "raw_ltl": "F & R F Z",
    "canonical_ltl": "finally ( and (  go to the red room , finally ( go to the green room with chair )  )  )",
    "eng": "go to the red room and push the chair into the green room"
}

translation_seeds.extend([additional_seed_1, additional_seed_2])

possible_decodings = {}
for seed in translation_seeds:
    canonical = seed['canonical_ltl']
    possible_decodings[canonical] = {
        'formula': canonical,
        'raw': seed['raw_ltl'],
    }

write_json(possible_decodings, "canonical.json")
process_data_points(data_points, translation_seeds)
