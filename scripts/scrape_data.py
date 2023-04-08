#!/usr/bin/env python3

import os
import requests
from bs4 import BeautifulSoup
import time
import random


def AzlyricsParseLyricsFromSongPage(
    url: str
) -> str:
    print(f"Parsing {url}")
    page = requests.get(url)
    time.sleep(10.0 + random.random()) # Give it at least 10 seconds before each page request so we don't get banned.
    soup = BeautifulSoup(page.content, "html.parser")
    lyrics_tags = soup.find_all("div", attrs={"class": None, "id": None})
    lyrics = [tag.getText() for tag in lyrics_tags]
    return ''.join(lyrics)

if __name__ == "__main__":
    # Get the main page for Notorious B.I.G's discography lyrics.
    website_url = "https://www.azlyrics.com/"
    discography_url = os.path.join(website_url, "n/notorious.html")
    discography_page = requests.get(discography_url)
    # Find links to every song page and parse the lyrics.
    soup = BeautifulSoup(discography_page.content, "html.parser")
    song_list_div = soup.find_all("div", class_="listalbum-item")
    all_lyrics = []
    for song_div in song_list_div:
        if song_div is None or song_div.find("a") is None:
            continue
        link = song_div.find("a").get("href")
        if website_url not in link:
            if link.startswith('/'):
                link = link[1:]
            link = os.path.join(website_url, link)

        lyrics = AzlyricsParseLyricsFromSongPage(link)
        all_lyrics.append(lyrics)
    all_lyrics = ''.join(all_lyrics)
    with open('notorious_lyrics.txt', 'w') as f:
        f.write(all_lyrics)

