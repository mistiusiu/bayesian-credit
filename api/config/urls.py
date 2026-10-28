"""URL map: ``/api/v1/`` (REST), ``/api/schema/`` (OpenAPI 3), ``/api/docs/`` (Swagger UI)."""

from django.contrib import admin
from django.urls import include, path
from django.views.generic import RedirectView
from drf_spectacular.views import SpectacularAPIView, SpectacularSwaggerView

from apps.common import views as common_views

urlpatterns = [
    path("", RedirectView.as_view(url="/api/docs/", permanent=False)),
    path("admin/", admin.site.urls),
    path("api/schema/", SpectacularAPIView.as_view(), name="schema"),
    path("api/docs/", SpectacularSwaggerView.as_view(url_name="schema"), name="swagger-ui"),
    path("api/v1/", common_views.api_root, name="api-root"),
    path("api/v1/", include("apps.common.urls")),
    path("api/v1/", include("apps.datasets.urls")),
    path("api/v1/", include("apps.fits.urls")),
    path("api/v1/", include("apps.simulations.urls")),
    path("api/v1/", include("apps.sentinel.urls")),
]

handler404 = "apps.common.views.not_found"
handler500 = "apps.common.views.server_error"
