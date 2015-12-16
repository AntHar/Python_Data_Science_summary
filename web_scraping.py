import requests #downloads webpage
from bs4 import BeautifulSoup #extract info (parse)

# download page
res = requests.get('https://www.google.com/search?q=' + ''.join(sys.argv[1:]) + '&tbm=nws')
content = res.content
# check if there was an error
try:
  res.raise_for_status()
except Exception as exc:
  print(' There was a problem: %s' % (exc))
  
# Extract info
parser = BeautifulSoup(content, 'html.parser') #or parse_marca=BeautifulSoup(res.text, "html.parser")

# I could just go inside things one by one:
head =  parser.head
title = head.title
title_text = title.text

# or use find all (BETTER!)
# Get a list of all occurences of the body tag in the element.
body = parser.find_all("body")
# Get the paragraph tag
p = body[0].find_all("p")
# Get the text
print(p[0].text)

# I could do search directly for p and and use ids:
second_paragraph_text = parser.find_all("p",id="second")[0].text
# same as:
second_paragraph_text = parser.find_all("p")[1].text


