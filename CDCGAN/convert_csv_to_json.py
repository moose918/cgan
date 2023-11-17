import json
import ast
import pandas as pd

import sys

# Check if at least one command-line argument is provided
if len(sys.argv) != 3:
    print("Usage: python script.py <src_file.csv> <dest_file.json>")
    exit(0)

# Access the command-line arguments starting from index 1 (0 is the script name)
src_file, dst_file = sys.argv[1:3]

# Print the arguments
print(f"Command-line arguments: {src_file} {dst_file}")

df = pd.read_csv(src_file)

try:
    levels = df['level']
except KeyError as e:
    levels = df.iloc[:, 0]
    
# Save the JSON data to the file
with open(dst_file, 'w') as file:
    file.write("[\n")

count = 1
size = len(levels)

for lvl in levels:
    json_lvl = json.dumps(ast.literal_eval(lvl))
    with open(dst_file, 'a') as file:
        file.write(json_lvl)

        if count == size:
            file.write("\n")
        else:
            file.write(",\n")
        count+=1
        print(f"wrote level {count}")

with open(dst_file, 'a') as file:
    file.write("]\n")

print(f'Saved {dst_file}')