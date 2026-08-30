from django.db import models

# home,score,away,agno
class partidos(models.Model):
    home = models.CharField(max_length=50)
    score = models.CharField(max_length=6)
    away = models.CharField(max_length=50)
    agno = models.CharField(max_length=4)

# home,score,away,year
class fixtures(models.Model):
    home = models.CharField(max_length=50)
    score = models.CharField(max_length=10)
    away = models.CharField(max_length=50)
    agno = models.CharField(max_length=4)

# pais,Pts,PJ,PG,PE,PP,GF,GC,Dif,Grupo,agno
class grupos(models.Model):
    pais = models.CharField(max_length=50)
    Pts = models.CharField(max_length=6)
    PJ = models.CharField(max_length=6)
    PG = models.CharField(max_length=6)
    PE = models.CharField(max_length=6)
    PP = models.CharField(max_length=6)
    GF = models.CharField(max_length=6)
    GC = models.CharField(max_length=6)
    Dif = models.CharField(max_length=6)
    Grupo = models.CharField(max_length=50)
    agno = models.CharField(max_length=6)

# agno, home, away, score_0, score_1, tournament
class clasificaciones(models.Model):
    agno = models.CharField(max_length=6)
    home = models.CharField(max_length=50)
    away = models.CharField(max_length=50)
    score_0 = models.CharField(max_length=10)
    score_1 = models.CharField(max_length=10)
    tournament = models.CharField(max_length=50)

# gkps, mds, mos, mms, VI, SOT, PKATT, PKATTALLOW
class proxy_desempegno(models.Model):
    gkps = models.CharField(max_length=50)
    mds = models.CharField(max_length=50)
    mos = models.CharField(max_length=50)
    mms = models.CharField(max_length=50)
    VI = models.CharField(max_length=50)
    SOT = models.CharField(max_length=50)
    PKATT = models.CharField(max_length=50)
    PKATTALLOW = models.CharField(max_length=50)