import numpy as np
import pandas as pd
import pickle
from fastapi import FastAPI, status, Request
from fastapi.responses import JSONResponse
from starlette.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
import joblib

app = FastAPI()
templates = Jinja2Templates(directory="public")
app.mount("/static", StaticFiles(directory="static"), name="static")

df = pd.read_csv('data.csv')

with open('tf_content.pkl', 'rb') as file:
    tf_content = pickle.load(file)

with open('tf_genre.pkl', 'rb') as file:
    tf_genre = pickle.load(file)

pop = pd.read_csv('pop_film.csv')

model_content = joblib.load('model_content.pkl')
model_genre = joblib.load('model_genre.pkl')

genres = ['Animation', 'Comedy', 'Family', 'Adventure', 'Fantasy', 'Romance', 'Drama', 'Action', 'Crime', 'Thriller', 'Horror', 'History', 'Science Fiction', 'Mystery', 'War', 'Foreign', 'Music', 'Documentary', 'Western', 'TV Movie']


def get_top10():
    pop['w_score'] = pop['w_score'].round(2)
    return pop[['title','w_score','release_date']]


def get_content_recommendations(movie_title, tfidf_matrix, model):
    movie_index = df[df['title'] == movie_title].index[0]
    movie_tfidf = tfidf_matrix[movie_index]
    _, indices = model.kneighbors(movie_tfidf, n_neighbors=100)

    recommended_movies = df[['title', 'w_score', 'release_date']].iloc[indices[0][1:]].drop_duplicates()
    recommended_movies = recommended_movies[recommended_movies['title'] != movie_title].head(10)

    return recommended_movies


def get_genre_recommendation(genre_name):
    genre_for_split = genre_name
    genres_list = genre_for_split.split(", ")
    genre_movies = df[df["genres"].apply(lambda x: all(genre in x for genre in genres_list))]
    genre_movies = genre_movies.drop_duplicates(subset='title')
    top_genre_movies = genre_movies.sort_values(by="w_score", ascending=False).head(10)
    top_genre_movies['w_score'] = top_genre_movies['w_score'].round(2)
    return top_genre_movies[['title', 'w_score', 'release_date']]

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
    films = get_content_recommendations(film_name, tf_content, model_content)
    films = films.values.tolist()
    return templates.TemplateResponse("rec_content.html", {"request": request, "films": films, "film_content": film_name})


@app.get("/genrefilm_rec")
async def genre2_recommendation(request: Request, film_name: str):
    if film_name not in df['title'].values:
        return JSONResponse(status_code=status.HTTP_400_BAD_REQUEST, content={"error": "Фильм не существует"})
    films = get_content_recommendations(film_name, tf_genre, model_genre)
    films = films.values.tolist()
    return templates.TemplateResponse("rec_genre.html", {"request": request, "films": films, "genres": genres, "film_genre": film_name})


@app.get('/api/film_suggestions')
def get_film_suggestions(film_name: str):
    film_name = film_name.lower()
    suggestions = df[df['title'].apply(lambda x: x.lower().startswith(film_name))]['title'].drop_duplicates().tolist()
    return {'suggestions': suggestions}
