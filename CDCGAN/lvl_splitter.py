import numpy as np
import json
import ast

with open("levels.json", "r") as file:
	levels = np.array(json.load(file))

i = -1
size = 100000
while True:
	i += 1
	dst_file = f"b{i}.json"

	batch_lvl = levels[i*size:(i+1)*size, :, :]
	if len(batch_lvl) == 0:
		break

	json_lvl = json.dumps(batch_lvl.tolist())
	with open(dst_file, 'w') as file:
		file.write(json_lvl)

	print(f'Saved {dst_file}')
	
	if i == -1:
		break
