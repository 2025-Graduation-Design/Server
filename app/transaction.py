from contextlib import contextmanager
from sqlalchemy.orm import Session

@contextmanager
def transactional_session(db: Session):
    """
    Spring Boot @Transactional처럼 사용할 수 있는 컨텍스트 매니저
    """
    try:
        yield db
        db.commit()
    except Exception as e:
        db.rollback()
        raise e
    finally:
        db.close()