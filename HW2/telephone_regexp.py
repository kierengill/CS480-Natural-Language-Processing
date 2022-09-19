import re

import fileinput

regex = '\(?[0-9]{3}?\)?[-]?\s?[0-9]{3}[-][0-9]{4}'

output = open("telephone_output.txt",'w')

for x in fileinput.input():
    for match in re.finditer(re.compile(regex), x):
        output.write(match.group() + '\n')