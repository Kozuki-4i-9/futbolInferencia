from django.contrib import admin
from .models import partidos, fixtures, grupos, clasificaciones

admin.site.register(partidos)
admin.site.register(fixtures)
admin.site.register(grupos)
admin.site.register(clasificaciones)