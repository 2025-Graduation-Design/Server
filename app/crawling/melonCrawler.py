from selenium import webdriver
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from webdriver_manager.chrome import ChromeDriverManager
import time

class MelonCrawler:
    BASE_URL = "https://www.melon.com/chart/index.htm"
    WEEKLY_CHART_URL = "https://www.melon.com/chart/week/index.htm?classCd={}"
    MONTHLY_CHART_URL = "https://www.melon.com/chart/month/index.htm?classCd={}"

    GENRE_CODES = {
        "ballad": "GN0100",
        "dance": "GN0200",
        "hiphop": "GN0300",
        "randb": "GN0400",
        "indie": "GN0500",
        "rock": "GN0600",
        "fork": "GN0800"
    }

    def __init__(self):
        self.options = webdriver.ChromeOptions()
        self.options.add_argument("headless")
        self.options.add_argument("log-level=3")
        self.service = Service(ChromeDriverManager().install())

    def crawl_songs(self, limit=100):
        """실시간 차트에서 상위 N곡 크롤링"""
        driver = webdriver.Chrome(service=self.service, options=self.options)
        driver.get(self.BASE_URL)
        time.sleep(2)

        songs = self._extract_songs_from_chart(driver, limit)
        driver.quit()
        return songs

    def crawl_genre_songs_weekly(self, genre="ballad", limit=100):
        """주간 장르 차트 크롤링"""
        if genre not in self.GENRE_CODES:
            return []

        url = self.WEEKLY_CHART_URL.format(self.GENRE_CODES[genre])
        driver = webdriver.Chrome(service=self.service, options=self.options)
        driver.get(url)
        time.sleep(2)

        songs = self._extract_songs_from_chart(driver, limit)
        driver.quit()
        return songs

    def crawl_genre_songs_monthly(self, genre="ballad", year_month="202503", limit=100):
        """월간 장르 차트 크롤링"""
        if genre not in self.GENRE_CODES:
            return []

        url = f"{self.MONTHLY_CHART_URL.format(self.GENRE_CODES[genre])}&moved=Y&rankMonth={year_month}"
        driver = webdriver.Chrome(service=self.service, options=self.options)
        driver.get(url)
        time.sleep(2)

        songs = self._extract_songs_from_chart(driver, limit)
        driver.quit()
        return songs

    def crawl_song_details(self, song_id):
        """곡 상세 페이지에서 가사, 장르 가져오기"""
        driver = webdriver.Chrome(service=self.service, options=self.options)
        song_url = f"https://www.melon.com/song/detail.htm?songId={song_id}"
        driver.get(song_url)
        time.sleep(2)

        # 가사
        try:
            more_button = driver.find_elements(By.CSS_SELECTOR, "button.lyricButton")
            if more_button:
                more_button[0].click()
                time.sleep(1)

            lyrics_text = driver.find_element(By.CSS_SELECTOR, "div.lyric").text.strip()
            lyrics_sentences = [line.strip() for line in lyrics_text.split("\n") if line.strip()]
        except:
            lyrics_sentences = []

        # 장르
        try:
            genre = ""
            dt_elements = driver.find_elements(By.CSS_SELECTOR, "dl.list dt")
            for i, dt in enumerate(dt_elements):
                if dt.text.strip() == "장르":
                    genre = driver.find_elements(By.CSS_SELECTOR, "dl.list dd")[i].text
                    break
        except:
            genre = ""

        driver.quit()
        return {"lyrics": lyrics_sentences, "genre": genre}

    def crawl_songs_with_details(self, limit=100):
        """실시간 차트에서 곡 + 가사 + 장르 크롤링"""
        songs = self.crawl_songs(limit)
        for song in songs:
            details = self.crawl_song_details(song["id"])
            song["lyrics"] = details["lyrics"]
            song["genre"] = details["genre"]
        return songs

    def crawl_songs_with_details_genre(self, genre, limit=100):
        """주간 장르별 곡 + 가사 + 장르 크롤링"""
        songs = self.crawl_genre_songs_weekly(genre=genre, limit=limit)
        for song in songs:
            details = self.crawl_song_details(song["id"])
            song["lyrics"] = details["lyrics"]
            song["genre"] = details["genre"]
        return songs

    def _extract_songs_from_chart(self, driver, limit):
        """차트에서 공통 song 정보 추출"""
        rows = driver.find_elements(By.CSS_SELECTOR, "tr.lst50") + driver.find_elements(By.CSS_SELECTOR, "tr.lst100")
        songs = []
        seen_ids = set()

        for row in rows[:limit]:
            try:
                song_id = row.find_element(By.CSS_SELECTOR, "input[type='checkbox']").get_attribute("value")
                if song_id in seen_ids:
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
                seen_ids.add(song_id)

            except Exception as e:
                print(f"오류 발생: {e}")
                continue

        return songs

    def crawl_songs_with_grouped_lyrics(self, limit=100, base_line=2, min_length=25):
        """실시간 차트에서 곡 + (가사 2줄 25자 이상 묶은 것) + 장르 크롤링"""
        songs = self.crawl_songs(limit)
        for song in songs:
            details = self.crawl_song_details_group(song["id"], base_line=base_line, min_length=min_length)
            song["lyrics"] = details["lyrics"]
            song["genre"] = details["genre"]
        return songs


    def crawl_song_details_group(self, song_id, base_line=2, min_length=25):
        """곡 상세 페이지에서 가사 (2줄 25자 이상 묶기), 장르 가져오기"""
        driver = webdriver.Chrome(service=self.service, options=self.options)
        try:
            song_url = f"https://www.melon.com/song/detail.htm?songId={song_id}"
            driver.get(song_url)
            time.sleep(2)

            try:
                more_button = driver.find_elements(By.CSS_SELECTOR, "button.lyricButton")
                if more_button:
                    more_button[0].click()
                    time.sleep(1)

                lyrics_text = driver.find_element(By.CSS_SELECTOR, "div.lyric").text.strip()
                lyrics_sentences = [line.strip() for line in lyrics_text.split("\n") if line.strip()]
                grouped_lyrics = self._group_lyrics_by_length(lyrics_sentences, base_line, min_length)
            except Exception as e:
                print(f"가사 크롤링 실패: {e}")
                grouped_lyrics = []

            try:
                genre = ""
                dt_elements = driver.find_elements(By.CSS_SELECTOR, "dl.list dt")
                for i, dt in enumerate(dt_elements):
                    if dt.text.strip() == "장르":
                        genre = driver.find_elements(By.CSS_SELECTOR, "dl.list dd")[i].text
                        break
            except Exception as e:
                print(f"장르 크롤링 실패: {e}")
                genre = ""

            return {"lyrics": grouped_lyrics, "genre": genre}
        finally:
            driver.quit()

    def _group_lyrics_by_length(self, lyrics_list, base_line=2, min_length=25):
        """가사를 base_line개 줄로 묶고, 25자 이상 될 때까지 추가하는 함수"""
        grouped = []
        i = 0
        while i < len(lyrics_list):
            chunk_lines = lyrics_list[i:i+base_line]
            chunk_text = " ".join(chunk_lines)
            j = i + base_line
            while len(chunk_text.replace("\n", "")) < min_length and j < len(lyrics_list):
                chunk_text += " " + lyrics_list[j]
                j += 1
            grouped.append(chunk_text.strip())
            i = j
        return grouped