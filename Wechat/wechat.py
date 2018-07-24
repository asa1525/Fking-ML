import ssl

from urllib.request import Request
from urllib.request import urlopen

context = ssl._create_unverified_context()

# HTTP 请求
requests = Request(url="https://foofish.net/pip.html",
                  method="GET",
                  headers={"Host": "foofish.net"},
                  data=None)

# HTTP 响应
response = urlopen(requests, context=context)
headers = response.info()  # 响应头
content = response.read() # 响应体
code = response.getcode() # 状态码...

r = requests.get("https://httpbin.org/ip")
print(r)
print(r.status_code)
print(r.content)