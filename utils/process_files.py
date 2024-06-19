import os
from collections import defaultdict
import pickle as pkl

root_path = '../data/deGraphCS/c'
folders = os.listdir(root_path)
contents = defaultdict(dict)

for folder in folders:
    print(folder)
    files = os.listdir(os.path.join(root_path, folder))
    for file in files:
        print(file)
        name = file[:-6]
        file_type = file[-6:-4]
        with open(os.path.join(root_path, folder, file), 'r') as f:
            content = f.read()
        contents[name][file_type] = content

with open('processed.pkl', 'wb') as f:
    pkl.dump(contents, f)

    