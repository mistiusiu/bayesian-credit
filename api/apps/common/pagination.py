"""DRF LimitOffset envelope ``{count, next, previous, results}`` (types.ts ``Paginated<T>``)."""

from rest_framework.pagination import LimitOffsetPagination


class DefaultPagination(LimitOffsetPagination):
    default_limit = 50
    max_limit = 500
