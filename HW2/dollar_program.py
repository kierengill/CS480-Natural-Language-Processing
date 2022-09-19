import re

import fileinput

regex = '\$\d*\.\d+|(\$)?(\d)+(\,)?(\d)+ dollars|a dollar|\$\d*\,\d+\.\d+|\$\d*\,?\d*\,?\d+\.?\d+|(\w+)\s+cents|(\w+)\s+thousand dollars?|(\w+)\s+hundred thousand dollars?|\w*ty\s+dollars|\w*teen\s+dollars|ten dollars|two dollars|three dollars|four dollars|five dollars|six dollars|seven dollars|eight dollars|nine dollars'

output = open("dollar_output.txt",'w')

for x in fileinput.input():
    for match in re.finditer(re.compile(regex), x):
        output.write(match.group() + '\n')

