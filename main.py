import numpy as np
import pandas as pd
import pickle
import uuid
from fastapi import FastAPI, Body, status, Request
from fastapi.responses import JSONResponse, FileResponse
from starlette.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi.templating import Jinja2Templates

app = FastAPI()
templates = Jinja2Templates(directory="public")
app.mount("/static", StaticFiles(directory="static"), name="static")

df = pd.read_csv('data.csv')
similarity_content_matrix = np.load('matrix_content_rating.npz')['arr']
similarity_genre_matrix = np.load('matrix_genres_rating.npz')['arr']
pop = pd.read_csv('pop_film.csv')

genres = ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie']


def get_top10():
    pop['w_score'] = pop['w_score'].round(2)
    return pop[['title','w_score','release_date']]

def get_content_recommendations(movie_title, similarity_matrix, n=10):
    movie_index = df[df['title'] == movie_title].index[0]

    similar_movies = list(enumerate(similarity_matrix[movie_index]))

    similar_movies_sorted = sorted(similar_movies, key=lambda x: x[1], reverse=True)

    recommended_movies = []
    recommended_rating = []
    recommended_release = []
    recommended_count = 0

    for index, similarity in similar_movies_sorted:
        if index != movie_index:
            recommended_movie_title = df.iloc[index]['title']
            recommended_movie_rating = df.iloc[index]['w_score']
            recommended_movie_release_date = df.iloc[index]['release_date']

            if recommended_movie_title != movie_title:
                if recommended_movie_title not in recommended_movies:
                    recommended_movies.append(recommended_movie_title)
                    recommended_rating.append(recommended_movie_rating)
                    recommended_release.append(recommended_movie_release_date)
                    recommended_count += 1

                    if recommended_count == n:
                        break

    films = []
    for i in range(len(recommended_movies)):
        films.append((recommended_movies[i], recommended_rating[i], recommended_release[i]))

    recommended_df = pd.DataFrame(films, columns=['title', 'rating', 'release_date'])
    recommended_df['rating'] = recommended_df['rating'].round(2)
    return recommended_df


def get_genre_recommendation(genre_name):
    genre_for_split = genre_name
    genres_list = genre_for_split.split(", ")
    genre_movies = df[df["genres"].apply(lambda x: all(genre in x for genre in genres_list))]
    genre_movies = genre_movies.drop_duplicates(subset='title')
    top_genre_movies = genre_movies.sort_values(by="w_score", ascending=False).head(10)
    top_genre_movies['w_score'] = top_genre_movies['w_score'].round(2)
    return top_genre_movies[['title', 'w_score', 'release_date']]Ё

@app.get("/")
async def root(request: Request):
    films = get_top10()
    films = films.values.tolist()
    return templates.TemplateResponse("index.html", {"request": request, "films": films[:10]})


@app.get("/genre")
async def genre(request: Request):
    return templates.TemplateResponse("rec_genre.html", {"request": request, "genres": genres})


@app.get("/genres_rec")
async def genres_recommendation(request: Request, genre: str):
    films = get_genre_recommendation(genre)
    films = films.values.tolist()
    return templates.TemplateResponse("rec_genre.html", {"request": request, "films": films[:10], "genres": genres, "genre": genre})


@app.get("/content")
async def genre(request: Request):
    return templates.TemplateResponse("rec_content.html", {"request": request})

@app.get("/content_rec")
async def content_recommendation(request: Request, film_name: str):
    if film_name not in df['title'].values:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Фильм не существует"})
    films = get_content_recommendations(film_name, similarity_content_matrix)
    films = films.values.tolist()
    return templates.TemplateResponse("rec_content.html", {"request": request, "films": films, "film_content": film_name})


@app.get("/genrefilm_rec")
async def genre2_recommendation(request: Request, film_name: str):
    if film_name not in df['title'].values:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Фильм не существует"})
    films = get_content_recommendations(film_name, similarity_genre_matrix)
    films = films.values.tolist()
    return templates.TemplateResponse("rec_genre.html", {"request": request, "films": films, "genres": genres, "film_genre": film_name})


@app.get('/api/film_suggestions')
def get_film_suggestions(film_name: str):
    film_name = film_name.lower()
    suggestions = df[df['title'].apply(lambda x: x.lower().startswith(film_name))]['title'].drop_duplicates().tolist()
    return {'suggestions': suggestions}
