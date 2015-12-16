import requests #downloads webpage
import bs4 #extract info

# download page
res = requests.get('https://www.google.com/search?q=' + ''.join(sys.argv[1:]) + '&tbm=nws')
# check if there was an error
try:
  res.raise_for_status()
except Exception as exc:
  print(' There was a problem: %s' % (exc))
