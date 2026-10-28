from django.urls import path
from rest_framework.routers import SimpleRouter

from . import views

router = SimpleRouter()
router.register("jobs", views.JobViewSet, basename="job")

urlpatterns = [
    path("health/", views.health, name="health"),
    path("metrics/", views.metrics, name="metrics"),
    *router.urls,
]
