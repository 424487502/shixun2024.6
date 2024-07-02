# Generated by Django 5.0.6 on 2024-06-25 00:24

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='User',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('uname', models.CharField(max_length=20)),
                ('upwd', models.CharField(max_length=50)),
                ('utel', models.CharField(max_length=50, null=True)),
                ('isvip', models.BooleanField(default=False)),
                ('isactive', models.BooleanField(default=True)),
            ],
            options={
                'db_table': 'userinfos',
            },
        ),
    ]
