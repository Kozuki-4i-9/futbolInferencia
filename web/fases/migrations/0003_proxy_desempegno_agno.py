from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('fases', '0002_proxy_desempegno'),
    ]

    operations = [
        migrations.AddField(
            model_name='proxy_desempegno',
            name='agno',
            field=models.CharField(default='', max_length=6),
        ),
    ]
