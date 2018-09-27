
# coding: utf-8

# In[9]:


import requests
from bs4 import BeautifulSoup
import re
import csv

'''
res = requests.get('https://en.wikipedia.org/wiki/Lists_of_American_films')
soup = BeautifulSoup(res.text)
y = 0 # only 2010~2017
for year in soup.find_all(href=re.compile("^/wiki/List_of_American_films_of_201\d")):
    print(year['href'])
    y = y + 1
    if y == 8 : break
'''

res = requests.get('https://en.wikipedia.org/wiki/List_of_American_films_of_2017')
soup = BeautifulSoup(res.text)
table = soup.find( "table", class_='wikitable sortable' )
FileName = soup.find('h1').string 
with open( FileName + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([ 'Movie', 'actor' ])
    for tr in table.find_all( "tr" ) :
        stage = 1
        movie_name = ""
        actor_list = []
        try:
            for td in tr.find_all( "td" ):
                if stage == 1 :
                    print( 'movie_name= ' + td.find("a").text )
                    movie_name = td.find("a").text
                    
                elif stage == 3 :
                    for actor in td.find_all("a"):
                        print( 'actor= ' + actor.text )
                        actor_list.append(actor.text)
                stage = stage + 1
            string = ""
            for i in range ( 0 ,len(actor_list) ):
                if i != len(actor_list) - 1 :
                    string = string + str(actor_list[i]) + ","
                else :
                    string = string + str(actor_list[i])
                    
            writer.writerow([ movie_name , string ])
            print( ' ')
        except:
            print( 'no element' )
            


# In[13]:


import requests
from bs4 import BeautifulSoup
import re
import csv

res = requests.get('https://en.wikipedia.org/wiki/List_of_American_films_of_2017')
soup = BeautifulSoup(res.text)
FileName = soup.find('h1').string 
with open( FileName + '.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    writer.writerow([ 'Movie', 'actor' ])
    for table in soup.find_all( "table", class_='wikitable sortable' ):
        for tr in table.find_all( "tr" ) :
            stage = 1
            movie_name = ""
            actor_list = []
            try:
                for td in tr.find_all( "td" ):
                    if stage == 1 :
                        print( 'movie_name= ' + td.find("a").text )
                        movie_name = td.find("a").text

                    elif stage == 3 :
                        for actor in td.find_all("a"):
                            print( 'actor= ' + actor.text )
                            actor_list.append(actor.text)
                    stage = stage + 1
                string = ""
                for i in range ( 0 ,len(actor_list) ):
                    if i != len(actor_list) - 1 :
                        string = string + str(actor_list[i]) + ","
                    else :
                        string = string + str(actor_list[i])

                writer.writerow([ movie_name , string ])
                print( ' ')
            except:
                print( 'no element' )

