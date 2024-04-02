# @FileName  :CallTest.py
# @Author    :632107110111_张永锐
# @Time      :2023/9/24 21:27

class Person:
    def __call__(self, name):
        print("__call__" + name)

    def hello(self, name):
        print("hello" + name)

person = Person()
person("zhangsan")
person.hello("lisi")