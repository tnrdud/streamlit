import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD

# CSV 파일을 불러오는 함수
def load_data(file_path, nrows=None):
    return pd.read_csv(file_path, nrows=nrows)

# 도서 추천 함수 (TruncatedSVD를 사용하여 차원 축소 적용)
def recommend_books(title, data, n_components=100):
    # 입력받은 도서 제목에 해당하는 책 찾기
    book = data[data['TITLE_NM'].str.contains(title, case=False, na=False)]
    if book.empty:
        return None, None
    
    # ISBN_ADITION 벡터화
    count_vectorizer = CountVectorizer(analyzer='char', ngram_range=(1, 3))
    isbn_matrix = count_vectorizer.fit_transform(data['SGVL_ISBN_ADTION_SMBL_NM'].astype(str))
    
    # Truncated SVD로 차원 축소
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    isbn_reduced = svd.fit_transform(isbn_matrix)
    
    # 코사인 유사도 계산 (차원 축소된 데이터로)
    cosine_sim = cosine_similarity(isbn_reduced, isbn_reduced)
    
    # 입력받은 책의 인덱스
    book_idx = book.index[0]
    
    # 유사도 점수 높은 순서로 정렬하여 추천 도서 5권 추출
    sim_scores = list(enumerate(cosine_sim[book_idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    
    book_title = book['TITLE_NM'].values[0]
    
    # 동일한 제목을 제외하고 유사한 도서 추출
    book_indices = [i[0] for i in sim_scores if data['TITLE_NM'].iloc[i[0]] != book_title]
    
    # 추천 도서 5권 추출
    recommended_books = data.iloc[book_indices[:5]]

    return book, recommended_books

# Streamlit 앱 구성
st.title('책책책 📚 책의 위치를 알려드리고, 유사 도서를 추천해드립니다😍')

# 도서관 데이터 로드 (전체 데이터 사용)
library_df = pd.read_csv(r"C:\Users\msy\Downloads\python\LIBRARY_202408 .csv")  # 도서관 데이터 로드
sido_options = library_df['ONE_AREA_NM'].unique()  # 시도 목록 추출

# 도서 데이터 로드 (행 수 제한 없이 전체 데이터 사용)
data = load_data(r'BOOK_PUB_202408.csv.xlsx.xlsx')

# 지역 선택 (시도와 시군구)
selected_sido = st.selectbox('시도를 선택하세요', ['전체'] + list(sido_options))

if selected_sido != '전체':
    selected_sigungu = library_df[library_df['ONE_AREA_NM'] == selected_sido]['TWO_AREA_NM'].unique()
    selected_sigungu = st.selectbox('시군구를 선택하세요', ['전체'] + list(selected_sigungu))

# 도서 제목 입력
title_input = st.text_input('도서 제목을 입력하세요')

# 입력한 도서 제목의 위치 출력 및 추천 도서 제공
if title_input:
    # 책의 위치 찾기
    book_matches = data[data['TITLE_NM'].str.contains(title_input, case=False, na=False)]
    
    if not book_matches.empty:
        # 리스트에서 선택할 수 있도록 책 제목 추출
        # book_titles = book_matches['TITLE_NM'].tolist()
        book_titles_with_authors = book_matches.apply(lambda row: f"{row['TITLE_NM']} by {row['AUTHR_NM']}", axis=1).tolist()
        selected_book_info = st.selectbox("검색된 책 목록에서 선택하세요", options=book_titles_with_authors)

        # 선택한 책의 정보를 가져오기
        selected_book_title = selected_book_info.split(" by ")[0]  # 선택한 제목
        selected_book = book_matches[book_matches['TITLE_NM'] == selected_book_title].iloc[0]

        # 선택한 책의 정보를 가져오기
        # selected_book = book_matches[book_matches['TITLE_NM'] == selected_book_title].iloc[0]

        # 책 정보를 표시
        book_title = selected_book['TITLE_NM']
        author_name = selected_book['AUTHR_NM']
        publication_year = selected_book['PBLICTE_YEAR']
        st.write(f"**검색하신 책:** {book_title} by {author_name} ({publication_year})")
        
        # 선택한 책이 있는 도서관 코드 추출
        library_codes = selected_book['LBRRY_CD']  # 도서관 코드 (해당 책이 있는 도서관 번호)

        # 도서관 코드에 해당하는 도서관 정보 찾기
        matched_libraries = library_df[library_df['LBRRY_CD'] == library_codes]

        # 선택한 지역에 따라 도서관 필터링
        if selected_sido != '전체':
            matched_libraries = matched_libraries[matched_libraries['ONE_AREA_NM'] == selected_sido]
            if selected_sigungu != '전체':
                matched_libraries = matched_libraries[matched_libraries['TWO_AREA_NM'] == selected_sigungu]

        # 지도 초기화 (임의의 시작 좌표, 예: 서울 기준)
        start_location = [35.157180, 129.062966]
        m = folium.Map(location=start_location, zoom_start=14)
        marker_cluster = MarkerCluster().add_to(m)

        # 일치하는 도서관 마커 추가
        for i, j, name, addr, tel in zip(
            matched_libraries["LBRRY_LA"], 
            matched_libraries["LBRRY_LO"], 
            matched_libraries["LBRRY_NM"], 
            matched_libraries["LBRRY_ADDR"], 
            matched_libraries["TEL_NO"]
        ):
            marker = folium.CircleMarker(
                location=[i, j],
                radius=10,
                color='green',
                fill=True,
                fill_color='green',
                fill_opacity=0.6,
                tooltip=f"<span style='font-size: 20px;'>{name}</span>"
            )

            # 팝업 추가
            popup_content = f"<div style='width: 200px; height: 100px; font-size: 18px;'>" \
                            f"<strong>{name}</strong><br>주소: {addr}<br>전화번호: {tel}" \
                            f"</div>"
    
            folium.Popup(popup_content).add_to(marker)  # 팝업을 마커에 추가
            marker.add_to(marker_cluster)  # 마커 클러스터에 추가

        # 지도 표시
        folium_static(m)

        # 유사 도서 추천
        st.write("### 유사한 도서 추천 목록")
        _, recommendations = recommend_books(selected_book_title, data.head(20000), n_components=100)  # 행수 제한하여 추천

        if recommendations is not None and not recommendations.empty:
            for idx, rec_book in recommendations.iterrows():
                st.write(f"- {rec_book['TITLE_NM']} by {rec_book['AUTHR_NM']}")
        else:
            st.write("추천할 유사 도서를 찾을 수 없습니다.")
    else:
        st.write('해당 도서 제목에 해당하는 책을 찾을 수 없습니다.')
