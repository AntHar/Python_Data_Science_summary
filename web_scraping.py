import requests #downloads webpage
import bs4 #extract info

res = requests.get('https://www.google.com/search?q=' + ''.join(sys.argv[1:]) + '&tbm=nws')
################ res.raise_for_status()
