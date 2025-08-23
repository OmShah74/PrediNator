from django.urls import path
from . import views

app_name = 'game_app'
urlpatterns = [
    path('play/', views.play_view, name='play'),
    path('answer/', views.answer_view, name='answer'),
    path('learn_feedback/', views.learn_feedback_view, name='learn_feedback'),
    path('process_learning/', views.process_learning_view, name='process_learning'),
    path('add_question_form/', views.add_question_form_view, name='add_question_form'),
    path('submit_new_question/', views.submit_new_question_view, name='submit_new_question'),
    path('reset/', views.reset_game_view, name='reset_game'),
]