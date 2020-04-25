## Real-time Personalization using Embeddings for Search Ranking at Airbnb

프로젝트 전 지식 채우기용 논문 리딩 후 요약 내용입니다.

### Table of Contents
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

### 임베딩이란?
범주형 자료를 연속형 벡터(Continuous Vector) 형태로 변환 시키는 것

- 장점 1. 차원 축소 2. 의미 도출이 쉬워짐
- 용도 1. 추천 시스템 2. 클러스터링 등

### Airbnb 도메인 이해
1. P2P 플랫폼 (Peer 2 Peer)

    Social network paradigm has moved to a network with 2 distinct types of participants representing supply & demand.

    EX. 숙박(에어비앤비), 라이드 셰어링 (우버, 리프트), 온라인 숍(Etsy)

    소비자와 공급자가 함께 참여하는 시장에서 추천 시스템을 구성하기 위해서는, 소비자(숙박 예약자)와 공급자(에어비앤비 호스트) 모두를 충족시킬 수 있는 추천 시스템과 Search Ranking 모델을 구성해야 한다.

    Need to optimize search results for both hosts and guests

    - 소비자는 자기 입맛에 맞는 검색 랭킹을 볼 수 있도록
        - 이 방식도 두 가지 방향을 병행하는데,
            1. Rank high listings that are appealing to the guest
            2. Detect lower listings that would likely reject the guest
    - 호스트는 (ex. 개인의 신뢰도에 따라 신뢰할 수 있는 소비자들) 선호하는 타입의 소비자들에게 자신의 상품을 노출 시키도록

2. 활용할 수 있는 데이터 구분
    - In-session signals

        모바일 세션 내에서의 본 리스팅을 얼마나 클릭, 호스트 연락 등

    - Negative signal

        상위 리스팅을 그냥 skip 했는지

3. 에어비앤비 유저 예약 패턴

    **— 같은 아이템을 두번 이상 소비하지 않는다. (컬테와 유사)**

    **— 하나의 리스팅은 시간대별로 하나의 게스트만 수용 가능하다.**

    - Proxy Signal, for short-term user interest

        리얼 타임 개인화 위해 즉각적인 유저 액션 (클릭 등) 을 임베딩

        — **한번 예약하기 위해 이것 저것 검색할 때에 단일 market 에 대한 검색으로만 한정된다 (ex. LA 여행 가면 LA 숙소만 찾는다) (컬테와 다름)**

        **— 즉, cross-market search X**

    - Sparse Signal, for long-term user interest

        리얼 타임은 아니지만 긴 타임프레임을 두고 유저 별로 예약한 내역들을 의미

        — **에어비앤비는 예약 1회로 끝난 유저가 long tail로 많다.**

        — **그리고 유저마다 예약 간의 Term 이 상당히 길다**

        → 이 부분은 User id 단위가 아닌 유저들을 묶어 User type 단위로 임베딩함으로 해결.

        → 유저를 묶는 단위는 many-to-one rule-based mapping

4. 에어비앤비 상품 특성
    - location, 가격대, architecture, style, ratings, room type (private, shared, entire apt), capacity 등

5. 이 논문에서 참고한 것

    주로 랭킹 엔진의 추천 시스템과 개인화 메소드를 참고 (Yahoo, Etsy, Criteo, LinkedIn, Tinder, Tumblr)

### 단기 추천: Listing Embeddings

목차

- 리스팅이 무엇인지, 어떤 수식, 어떤 가정이 들어갔는지
- Cold start 리스팅 임베딩
- 리스팅 임베딩 간의 거리 계산
- 한계점

1. 리스팅이란

    ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e294eafe-b32b-4788-bfe4-d9f74876df65/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/e294eafe-b32b-4788-bfe4-d9f74876df65/Untitled.png)

    상품 하나하나 의미하는 듯.

    이 리스팅을 임베딩하는 것이 이하 내용

2. 수식과 가정
    - 세션 하나 S = (Listing 1 ~ M)
    - Listing 1 ~ M 은 Uninterrupted Sequence that were clicked by User
    - 세션은 각 클릭 간에 30분 이상의 term 이 생기면 다른 세션으로 구분한다.
    - 얻고자 하는 값: 각 리스팅의 Real-valued representation 점수

        사용 모델: Skip-gram

        ![https://s3-us-west-2.amazonaws.com/secure.notion-static.com/46bf144b-228f-4779-b113-7cf6042d9f06/Untitled.png](https://s3-us-west-2.amazonaws.com/secure.notion-static.com/46bf144b-228f-4779-b113-7cf6042d9f06/Untitled.png)

        메인 변수: 리스팅별로, Click-based Value( 예상컨데 0,1 인듯) 로 채워진 벡터 V

        추가 파라미터: forward looking, backward looking for CONTEXT

    - 전체 리스팅 간의 Context 조합을 구하려면, 연산 양이 터짐

        → Negative Sampling Approach, computational complexity를 줄여주는 방식

        수식: Argmax Positive Pair (Listings clicked, Context Sequence) + Negative Pair (Listings clicked, n Randomly sampled listings from entire listings)

        특징: Negative random sampling 을 할 때, 이왕이면 Positive pair 과는 다른 마켓에서 샘플링되도록 한다! 

        모델: Stochastic gradient ascent

    - Global context 추가; 그 리스팅 검색에서 예약까지 간 애들은 특별하게 모델에서 예외로 고려해준다. 즉 **session을 exploratory X booked 두 가지로 구분하는 방향**

        Adding booked listing as global context, such that it will be always predicted no matter if it is within the context window or not

        그 방식은: 논문을 더 깊이 읽어보는게 좋겠다.

3. 콜드 스타트

    새롭게 추가된 리스팅(집 상품)의 콜드 스타트를 방지하는 방법

    - 호스트가 등록 시에 location, price, listing type(private room 등) 등록함
    - geographical 거리 상 가까운 & 위에서 등록한 피쳐에 해당되는 리스팅 3개의 임베딩을 가져 온다
    - 그 3개 임베딩의 mean vector 가져와서 임베딩 생성
4. 리스팅 임베딩 평가 방법
    1. 디멘션은 32
    2. K means clustering 
    3. Evaluate average cosine similarities 
5. 임베딩으로 변환이 불가능한 상품 피쳐가 있다.
    - architecture
    - style
    - feel

    이런 정성적인 피쳐는 airbnb 에서 내부에서 디벨롭한 툴로.

    [https://www.youtube.com/watch?v=1kJSAG91TrI&feature=youtu.be](https://www.youtube.com/watch?v=1kJSAG91TrI&feature=youtu.be)

### 장기 추천: User-type Embeddings & Listing-type Embeddings

위에서 다룬 단기 추천은 같은 시장 내에서 (같은 지역) 의 리스팅 간의 유사도 측정에 효과적

장기 추천은 보다 다른 시장 간의 리스팅 간 유사도 측정에 효과적 (cross-market similarities)

유저 기준의 예약 시퀀스를 활용함.

1. 로직
    - booking session Si = Sequence of listing booked by user j

    — 적어도 리스팅이 5 ~ 10 개는 되어야 함

    — 리스팅 간에 time interval 이 길면 그 새에 유저들의 preference 변화할 가능성 높음

    — 양 자체가 적음. 예약이

    - 위 3가지 문제 해결 위해 user id 들을 user type 단위로 묶어서 벡터 생성.
    - 리스팅도 각 상품 하나씩뿐이니 listing id 묶어서 listing type 단위로 묶는다.

        그 묶는 방식은? rule-based mapping.

        그냥 피쳐 조합별로 묶는다.

        sparse matrix 해결 위함

    - 세션을 (리스팅 타입, 유저 타입) 튜플의 시퀀스로 구성함
2. 콜드 스타트
    - User type 에 예약 내역이 없어도, 초기에 등록하는
        - 시장, 언어, 디바이스 종류, 프로파일 등록 정도 여부

    요것을 기반으로 평균적인 예약 패턴으로 채워 넣음

3. 형태 이런 식

    유저 타입별로

    [user type of LA, low 지불용의, 프로파일 등록 여부, 등등등](https://www.notion.so/b37da63a00734c008b1e517e56ed5c07)

### 리얼 타임 personalization

- 위 1,2번이 리얼 타임으로 implement 되는 방법

    d = 32

    에어플로우 활용

- 학습 방식

                             1단                                2단

    Data → (기존) listing algorithm    → K -nearest neighbor

            (논문) embedding-based 

    Pairwise Regression

    Gradient Boosting Decision Tree

    Hold out method of 80,20 비율

    inclusive method로 신규 데이터 학습할 때 30일간의 데이터 모아서 새로 모델을 학습하는 방식 (기존 모델에 추가 학습이 아님)

- 수식, 가정 -

    리스팅 하나 하나마다 Feature vector, Reaction vector 의 튜플로 구성됨

    Set D = (listing feature, results by the list) 라고 result 를 리스트 올린 후 일주일 기다려 수집해서 쌓음

- 어떤 모델을 사용했고, 어떤 feature를 넣어 모델 scoring 을 도왔는지
    - features 종류
    1. Listing features : price per night 상품의 피쳐
    2. Query features : 검색 시에 피쳐
    3. User features: 유저 예약 정보
    4. Cross features: 위의 123 피쳐를 엮어 서 derive한 피쳐


## Reference
[Airbnb Engineering Blog](https://medium.com/airbnb-engineering/listing-embeddings-for-similar-listing-recommendations-and-real-time-personalization-in-search-601172f7603e)
