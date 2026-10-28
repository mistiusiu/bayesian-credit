from rest_framework.routers import SimpleRouter

from .views import SimulationViewSet

router = SimpleRouter()
router.register("simulations", SimulationViewSet, basename="simulation")
urlpatterns = router.urls
