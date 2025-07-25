from enum import Enum, auto

class InsertionMode(Enum):
    DECREASE_ONLY = "decrease_only"  # 길이 감소하는 경우만 삽입
    INSERT_WITHOUT_OPTIMIZE = "insert_without_optimize"  # 길이 증가해도 삽입, 최적화 없음
    INSERT_WITH_OPTIMIZE = "insert_with_optimize"  # 길이 증가해도 삽입 + 지역 최적화