import smtplib, ssl, _io
import gin

def send(user, password, message, to_addrs=None, subject=None):
  context = ssl.create_default_context()
  with smtplib.SMTP_SSL("smtp.gmail.com", 465, context=context) as server:
    server.login(user, password)
    if '@' not in user: 
      user+='@gmail.com'
    if to_addrs is None:
      to_addrs = user
    if subject is not None:
      message = 'Subject: '+subject+'\n\n'+message
    server.sendmail(user, to_addrs, message)

@gin.configurable
def send_file(file, **kwargs):
  if isinstance(file, _io.TextIOWrapper):
    message = file.read()
  else:
    with open(file) as f:
      message = f.read()
  send(message=message, **kwargs)