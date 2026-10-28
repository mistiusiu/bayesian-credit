from rest_framework.routers import SimpleRouter

from .views import FitViewSet

router = SimpleRouter()
router.register("fits", FitViewSet, basename="fit")
urlpatterns = router.urls
