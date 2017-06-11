import os
import json

path = '../dataset/patch_1/'
template = {}
files = (file for file in os.listdir(path)
         if os.path.isfile(os.path.join(path, file)))


filesSorted = sorted(files, key=lambda x: int(x.split('_')[0]))

for filename in filesSorted:
        template[filename] = 0;

with open('scores_template_patch.json', 'w') as outfile:
    json.dump(template, outfile, indent=4, sort_keys=True)