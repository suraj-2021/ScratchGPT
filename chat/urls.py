
from django.urls import path
from chat import views

app_name = "chat"

urlpatterns = [
    path("",            views.index,               name="index"),      # Main chat page
    path("api/chat/",   views.chat_api,             name="chat_api"),   # AJAX chat endpoint
    path("api/reset/",  views.reset_conversation,   name="reset"),      # Clear history
]