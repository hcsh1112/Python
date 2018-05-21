
# coding: utf-8

# In[ ]:


import requests
from bs4 import BeautifulSoup
import pymysql  
from apscheduler.schedulers.blocking import BlockingScheduler
import datetime

def job() :
    res = requests.get('https://tw.appledaily.com/hot/daily')
    soup = BeautifulSoup(res.text)
    for news in soup.select('li'):
        for content in news.select('.aht_title'):
            resNew = requests.get(content.a.get('href'))
            soupNew = BeautifulSoup(resNew.text)
            try:
                href = str( content.a.get('href') )   #href
                title = soupNew.find('h1').text          #title
                count = soupNew.find(class_='ndArticle_view').text    #count
                date = soupNew.find(class_='ndArticle_creat').text.split("：")[-1]
                date = date.split('/')[0]+date.split('/')[1]+date.split('/')[2]        #date
                for img in soupNew.select('img'):
                    if img.get('src').split(":")[0] == 'https':
                        imghref = str(img.get('src'))        #imghref
                        break
            except:
                break
            # sqlconnect
            db = pymysql.connect("localhost", "root", "10602141", "applenews",  charset='utf8')
            cursor = db.cursor()
            cursor.execute('SELECT COUNT(id) FROM applenews.news;')
            total = cursor.fetchone()
            try:
                cursor.execute("INSERT INTO applenews.news( id, NewsTitle, NewsCount, NewsUrl, NewsDate, NewsImg) VALUES( %d, '%s', %d, '%s', %d, '%s');" % ( int(total[0]), title, int(count), href, int(date), imghref ))
                db.commit()
            except:
                db.rollback()

            db.close()
            
    print( 'done!!  at ' + datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

sched = BlockingScheduler()
sched.add_job(job, 'interval', hours=24 )
sched.start()

            

