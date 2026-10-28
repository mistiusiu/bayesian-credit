from rest_framework.routers import SimpleRouter

from .views import ReviewViewSet

router = SimpleRouter()
router.register("sentinel/reviews", ReviewViewSet, basename="review")
urlpatterns = router.urls
