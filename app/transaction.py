from contextlib import contextmanager
from sqlalchemy.orm import Session

@contextmanager
def transactional_session(db: Session):
    """
    Spring Boot @Transactional처럼 사용할 수 있는 컨텍스트 매니저
    """
    try:
        yield db  # ✅ 트랜잭션 시작
        db.commit()  # ✅ 정상 실행 시 커밋
    except Exception as e:
        db.rollback()  # 🚨 예외 발생 시 롤백
        raise e
    finally:
        db.close()  # ✅ 세션 닫기