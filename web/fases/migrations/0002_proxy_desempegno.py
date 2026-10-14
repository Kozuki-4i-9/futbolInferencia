# Escrita a mano el 2026-10-12 (makemigrations no corre sin conexion a la DB porque modulo.py la consulta al importarse)

from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fases', '0001_initial'),
    ]

    operations = [
        migrations.CreateModel(
            name='proxy_desempegno',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('pais', models.CharField(max_length=50)),
                ('gkps', models.CharField(max_length=50)),
                ('mds', models.CharField(max_length=50)),
                ('mos', models.CharField(max_length=50)),
                ('mms', models.CharField(max_length=50)),
                ('VI', models.CharField(max_length=50)),
                ('SOT', models.CharField(max_length=50)),
                ('PKATT', models.CharField(max_length=50)),
                ('PKATTALLOW', models.CharField(max_length=50)),
            ],
        ),
    ]
