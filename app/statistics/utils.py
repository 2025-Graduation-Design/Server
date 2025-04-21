import random
from collections import Counter

emotion_group_mapping = {
    1: "positive_active", 2: "positive_active",
    3: "positive_calm",   4: "positive_calm",
    5: "negative_flat",   6: "negative_flat",
    7: "negative_deep",   8: "negative_deep"
}

emotion_messages = {
    "positive_active": {
        "messages": [
            "이번 달은 정말 활기가 넘쳤어요!",
            "에너지가 뿜뿜한 한 달이었네요.",
            "자기 자신이 잘 달리고 있다는 증거예요!"
        ],
        "suggestions": [
            "지금의 기운으로 새로운 도전, 어떤가요?",
            "소중한 순간들을 사진이나 글로 남겨보세요.",
            "이 에너지를 주위에도 나눠보는 건 어때요?"
        ]
    },
    "positive_calm": {
        "messages": [
            "잔잔하고 만족스러운 시간이 많았어요.",
            "감정이 안정되고 평온했던 한 달이네요.",
            "바쁘지 않더라도 충실한 나날을 보내셨어요."
        ],
        "suggestions": [
            "일상의 감사를 글로 남겨보세요.",
            "이 고요함을 방해하지 않게 조용한 취미를 가져보는 건 어때요?",
            "명상이나 따뜻한 차 한잔으로 하루를 마무리해봐요."
        ]
    },
    "negative_flat": {
        "messages": [
            "조금 허무하거나 무기력한 시간이 많았던 것 같아요.",
            "감정이 납작하게 가라앉은 느낌이 들었던 한 달이네요.",
            "조금은 힘 빠진 나날들이었을 수 있어요."
        ],
        "suggestions": [
            "햇볕을 쬐며 산책하는 것부터 시작해봐요.",
            "큰 변화보다 사소한 루틴이 도움될 수 있어요.",
            "너무 애쓰지 않아도 돼요. 쉬어도 괜찮아요."
        ]
    },
    "negative_deep": {
        "messages": [
            "감정의 깊이가 깊었던 한 달이에요.",
            "무언가 마음 속을 뒤흔든 일이 있었을지도 몰라요.",
            "마음이 무겁고 복잡한 시간이었네요."
        ],
        "suggestions": [
            "지금 느낀 감정을 정리해보는 글쓰기를 추천해요.",
            "위로받을 수 있는 콘텐츠를 찾아보는 것도 좋아요.",
            "자기 자신을 탓하지 말고, 천천히 숨 고르기를 해봐요."
        ]
    }
}

def summarize_emotions(emotion_ids: list[int]):
    groups = [emotion_group_mapping[e] for e in emotion_ids if e in emotion_group_mapping]
    counter = Counter(groups)
    top_group = counter.most_common(1)[0][0]

    message = random.choice(emotion_messages[top_group]["messages"])
    suggestion = random.choice(emotion_messages[top_group]["suggestions"])

    return {
        "top_emotion_group": top_group,
        "group_distribution": dict(counter),
        "message": message,
        "suggestion": suggestion
    }