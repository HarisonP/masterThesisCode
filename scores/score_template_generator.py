import os
import json

path = '../dataset/'
template = {}
files = (file for file in os.listdir(path)
         if os.path.isfile(os.path.join(path, file)))


filesSorted = sorted(files, key=lambda x: int(x.split('_')[0]))
print(filesSorted)
for filename in filesSorted:
        template[filename] = 0;

with open('scores_template.txt', 'w') as outfile:
    json.dump(template, outfile, indent=4, sort_keys=True)