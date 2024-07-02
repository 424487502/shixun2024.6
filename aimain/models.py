from django.db import models
# Create your models here.
class User(models.Model):
    uname = models.CharField(max_length=20,verbose_name='姓名')   # 字段名 字符串
    upwd = models.CharField(max_length=50,verbose_name='密码')
    utel = models.CharField(max_length=50,null=True,verbose_name='电话')  # 允许为空
    isvip = models.BooleanField(default=False,verbose_name='是否会员')
    isactive = models.BooleanField(default=True,verbose_name='是否激活')

    def __str__(self):
        return self.uname   # 修改后台展示名

    class Meta:
        db_table = "userinfos"  # 修改表名
        verbose_name = "用户信息"
        verbose_name_plural = "用户信息"