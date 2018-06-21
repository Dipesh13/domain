# encoding=utf8
import requests
from bs4 import BeautifulSoup
import os

sport_name = ['football']
month = '2018/may/'
days = [i for i in range(1,31)]

link_count = 0
for s_name in sport_name:
    download_folder = os.path.join(os.getcwd(), s_name)
    if not os.path.exists(download_folder):
        os.makedirs(download_folder)
    for day in days:
        url = 'https://www.theguardian.com/'+s_name+'/'+month+str(day)+'/all'
        r = requests.get(url, allow_redirects=False)

        html = r.text
        soup = BeautifulSoup(html, "html5lib")

        links = []
        for a in soup.find_all('a', href=True):
            link_test = a['href'].split('/')
            if '2018' in link_test and 'may' in link_test and str(day) in link_test:
                links.append(a['href'])

        for link in links:
            res = requests.get(link, allow_redirects=False)
            html1 = res.text
            soup1 = BeautifulSoup(html1, "html5lib")
            download_path = os.path.join(download_folder, os.path.basename(link))
            file = open(download_path+'.txt','wb')
            for p in soup1.find_all('p'):
                file.write(p.text.encode('utf-8'))
            file.close()
            print("Completed {}".format(link))
            link_count+=1

print("Total Number of articles extracted: {}".format(link_count))