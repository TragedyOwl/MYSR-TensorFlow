import datetime
import utils

now_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S') + ": "

print(now_time)

utils.log_message('../log/test.log', "a", "start")

