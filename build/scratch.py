from datetime import datetime


str1 = datetime.now()
dt_string  = str1.strftime("%d_%m_%Y_%H_%M_%S_%f")[0:21]
print(dt_string)