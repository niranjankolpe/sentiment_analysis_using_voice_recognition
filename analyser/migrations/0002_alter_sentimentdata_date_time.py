# Generated by Django 4.2.3 on 2023-09-01 04:56

import datetime
from django.db import migrations, models


class Migration(migrations.Migration):

    dependencies = [
        ('analyser', '0001_initial'),
    ]

    operations = [
        migrations.AlterField(
            model_name='sentimentdata',
            name='date_time',
            field=models.DateTimeField(default=datetime.datetime(2023, 9, 1, 10, 26, 34, 472206)),
        ),
    ]
