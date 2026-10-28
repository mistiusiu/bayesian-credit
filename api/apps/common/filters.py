"""django-filter helpers: comma-separated ``?status=queued,running`` style filters."""

import django_filters

from .models import Job


class CharInFilter(django_filters.BaseInFilter, django_filters.CharFilter):
    pass


class NumberInFilter(django_filters.BaseInFilter, django_filters.NumberFilter):
    pass


class JobFilter(django_filters.FilterSet):
    status = CharInFilter(field_name="status", lookup_expr="in")
    kind = CharInFilter(field_name="kind", lookup_expr="in")
    target_type = django_filters.CharFilter(field_name="target_type")
    target_id = django_filters.NumberFilter(field_name="target_id")

    class Meta:
        model = Job
        fields = ["status", "kind", "target_type", "target_id"]
