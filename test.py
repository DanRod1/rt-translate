import re

string = '/tmp/chunk1.mp3'
pattern = '\d+'

# maxsplit = 1
# split only at the first occurrence
result = re.search(pattern=pattern,string=string) 
print(result.group())