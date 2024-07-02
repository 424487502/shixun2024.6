from django.contrib import admin

# Register your models here.
from .models import *
# 高级管理类
class UserAdmin(admin.ModelAdmin):
    list_display = ('uname','upwd','utel')  # 后台展示的字段
    list_editable = ('utel',)
    list_filter = ('isactive','isvip')
    search_fields = ('uname',)

admin.site.register(User,UserAdmin)