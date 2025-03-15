from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

class MelonCrawler:
    BASE_URL = "https://www.melon.com/chart/index.htm"  # 멜론 실시간 차트 URL
    WEEKLY_CHART_URL = "https://www.melon.com/chart/week/index.htm?classCd={}" # 주간 장르 URL

    GENRE_CODES = {
        "ballad": "GN0100",
        "dance": "GN0200",
        "hiphop": "GN0300",
        "randb": "GN0400",
        "indie": "GN0500",
        "rock": "GN0600"
    }

    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("headless")  # 화면 없이 실행
        self.options.add_argument("log-level=3")  # 불필요한 로그 제거
        self.service = Service(ChromeDriverManager().install())

    def crawl_songs(self, limit=100):
        """멜론에서 상위 N개의 노래 크롤링"""
        driver = webdriver.Chrome(service=self.service, options=self.options)
        driver.get(self.BASE_URL)
        time.sleep(2)  # 페이지 로딩 대기

        songs = []
        rows = driver.find_elements(By.CSS_SELECTOR, "tr.lst50") + driver.find_elements(By.CSS_SELECTOR, "tr.lst100")

        for row in rows[:limit]:
            try:
                song_id = row.find_element(By.CSS_SELECTOR, "input[type='checkbox']").get_attribute("value")
                song_name = row.find_element(By.CSS_SELECTOR, "div.rank01 > span > a").text
                artist_name = row.find_element(By.CSS_SELECTOR, "div.rank02 > a").text
                album_name = row.find_element(By.CSS_SELECTOR, "div.rank03 > a").text
                album_id = row.find_element(By.CSS_SELECTOR, "div.rank03 > a").get_attribute("href").split("'")[1]
                album_image = row.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
                uri = f"https://www.melon.com/song/detail.htm?songId={song_id}"

                song_info = {
                    "id": song_id,
                    "song_name": song_name,
                    "artist_name_basket": [artist_name],
                    "album_id": album_id,
                    "album_name": album_name,
                    "album_image": album_image,
                    "uri": uri
                }
                songs.append(song_info)
            except Exception as e:
                print(f"오류 발생: {e}")

        driver.quit()
        return songs

    def crawl_weekly_genre_songs(self, genre="ballad", limit=100):
        """🎵 주간 인기곡 크롤링 (장르별) - 원하는 곡 개수까지 크롤링 가능"""
        if genre not in self.GENRE_CODES:
            print(f"잘못된 장르 선택: {genre}")
            return []

        genre_code = self.GENRE_CODES[genre]
        url = self.WEEKLY_CHART_URL.format(genre_code)

        songs = []
        song_ids = set()
        driver = webdriver.Chrome(service=self.service, options=self.options)
        driver.get(url)
        time.sleep(2)

        rows = driver.find_elements(By.CSS_SELECTOR, "tr.lst50") + driver.find_elements(By.CSS_SELECTOR, "tr.lst100")

        for row in rows[:limit]:
            try:
                song_id = row.find_element(By.CSS_SELECTOR, "input[type='checkbox']").get_attribute("value")
                if song_id in song_ids:
                    continue

                song_name = row.find_element(By.CSS_SELECTOR, "div.rank01 > span > a").text
                artist_name = row.find_element(By.CSS_SELECTOR, "div.rank02 > a").text
                album_name = row.find_element(By.CSS_SELECTOR, "div.rank03 > a").text
                album_id = row.find_element(By.CSS_SELECTOR, "div.rank03 > a").get_attribute("href").split("'")[1]
                album_image = row.find_element(By.CSS_SELECTOR, "img").get_attribute("src")
                uri = f"https://www.melon.com/song/detail.htm?songId={song_id}"

                song_info = {
                    "id": song_id,
                    "song_name": song_name,
                    "artist_name_basket": [artist_name],
                    "album_id": album_id,
                    "album_name": album_name,
                    "album_image": album_image,
                    "uri": uri
                }
                songs.append(song_info)
                song_ids.add(song_id)

            except Exception as e:
                print(f"오류 발생: {e}")

        driver.quit()
        return songs

    def crawl_song_details(self, song_id):
        """특정 노래 ID로 장르, 가사 크롤링"""
        driver = webdriver.Chrome(service=self.service, options=self.options)
        song_url = f"https://www.melon.com/song/detail.htm?songId={song_id}"
        driver.get(song_url)
        time.sleep(2)

        try:
            more_button = driver.find_elements(By.CSS_SELECTOR, "button.lyricButton")
            if more_button:
                more_button[0].click()
                time.sleep(1)

            lyrics = driver.find_element(By.CSS_SELECTOR, "div.lyric").text.replace("\n", " ")
        except:
            lyrics = ""

        try:
            genre = ""
            dt_elements = driver.find_elements(By.CSS_SELECTOR, "dl.list dt")

            for i, dt in enumerate(dt_elements):
                if dt.text.strip() == "장르":
                    genre = driver.find_elements(By.CSS_SELECTOR, "dl.list dd")[i].text
                    break

        except Exception as e:
            print(f" 장르 가져오기 실패: {e}")
            genre = ""

        driver.quit()

        return {"lyrics": lyrics, "genre": genre}

    def crawl_songs_with_details(self, limit=100):
        """노래 목록과 가사, 장르 함께 크롤링"""
        songs = self.crawl_songs(limit)

        for song in songs:
            details = self.crawl_song_details(song["id"])
            song["lyrics"] = details["lyrics"]
            song["genre"] = details["genre"]

        return songs

    def crawl_songs_with_details_genre(self, genre, limit=100):
        """특정 장르의 곡을 크롤링하고, 가사 및 장르 정보를 추가"""
        songs = self.crawl_weekly_genre_songs(genre=genre, limit=limit)

        for song in songs:
            details = self.crawl_song_details(song["id"])
            song["lyrics"] = details["lyrics"]
            song["genre"] = details["genre"]

        return songs