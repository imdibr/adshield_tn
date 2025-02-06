import requests 
from bs4 import BeautifulSoup

#now lets creat another function to split our information like following and follower
def parse(data):
    res=data.split("-")[0]
    re=res.split(" ")

    #now lets creat a dictonary to store information
    inf={}
    inf["followers"]=re[0]
    inf["following"]=re[2]
    inf["post"]=re[4]
    return inf

#now lets creat function to scrap our information
def scrap(user):
    url="https://www.instagram.com/{}".format(user)

    r=requests.get(url)
    bs=BeautifulSoup(r.text,"html.parser")

    rep=bs.find("meta",property="og:description").attrs["content"]
    meta_tag = bs.find("meta", property="og:description")
    if meta_tag and 'content' in meta_tag.attrs:
        # Extract the content of the meta tag
        profile_description = meta_tag.attrs["content"]
        print("Profile Description:", profile_description)
    else:
        print("Meta tag with 'og:description' not found.")

    return parse(rep)


#now lets creat our main function to see result
if __name__ == "__main__":
    user=input("user name \t")
    print(scrap(user))

