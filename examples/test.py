import gin

@gin.configurable
def f(a=[0],b=None):
  for i in a:
    print(i)
  if b:
    print(b)

if __name__=='__main__':
  gin.parse_config_file('test.gin')
  f()
