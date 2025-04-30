from youtube_comment_downloader import YoutubeCommentDownloader
import pandas as pd

# 유튜브 영상 URL
video_url = "https://www.youtube.com/watch?v=4rgJCRGikW0"

# 댓글 수집
downloader = YoutubeCommentDownloader()
comments = []

# 정렬 방식은 정수로: 0 = 인기순, 1 = 최신순
for idx, comment in enumerate(downloader.get_comments_from_url(video_url, sort_by=0)):
    if idx >= 30:
        break
    comments.append({
        '댓글': comment['text']
    })

# 데이터프레임으로 저장
df = pd.DataFrame(comments)
df.to_excel("youtube_comments_tearscouple.xlsx", index=False)

print("🎉 30개 댓글 크롤링 완료! youtube_comments.xlsx 파일로 저장됨.")