## Real-time Personalization using Embeddings for Search Ranking at Airbnb

프로젝트 전 지식 채우기용 논문 리딩 후 요약 내용입니다.

## Table of Contents
- 임베딩 간략하게 설명
- AirBnb 도메인 특징, 추천 시스템에 어떻게 고려되어야 하는지

    → 컬테에 어느 부분을 반영하고 customize 하면 좋을 지

- 크게 단기 추천 알고리즘, 장기 추천 알고리즘, Real-time으로 추천에 대한 유저 반응을 로그 데이터 기반으로 반영하여 모델 성능 개선하는 방식 3가지로 구성됨.
    1. 단기 추천 알고리즘
        - Listing Embeddings
            - 리스팅이 무엇인지, 어떤 수식, 어떤 가정이 들어갔는지
            - Cold start 리스팅 임베딩
            - 리스팅 임베딩 간의 거리 계산
            - 한계점
    2. 장기 추천 알고리즘
        - User-type & Listing-type Embeddings
            - 유저 타입, 리스팅 타입이 무엇인지, 수식과 가정들
            - sparse-matrix 가 다수 산출되는 에어비앤비 도메인에 맞게 사용된 임베딩 방법
            - Cold start 방지 방법
            - 학습 방식
    3. Real-time Personalization in Search ( Search Ranking Model of Airbnb)
        - 위 1,2번이 리얼 타임으로 implement 되는 방법
        - 학습 방식
        - 수식, 가정 - 리스팅 하나 하나마다 Feature vector, Reaction vector 의 튜플로 구성됨
        - 어떤 모델을 사용했고, 어떤 feature를 넣어 모델 scoring 을 도왔는지
        - 결과
- 컬테에 어떤 부분을 활용하면 좋을 지

##


## Reference
[Airbnb Engineering Blog](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e)
