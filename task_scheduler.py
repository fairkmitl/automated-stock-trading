import schedule
import time
from portfolio import getPort
from db import insert

def job1():
    print("Job 1 is running...")
    port_detail = getPort()
    print(port_detail)
    # insert(port_detail)

def job2():
    print("Job 2 is running...")

# Schedule the jobs
schedule.every(5).seconds.do(job1)
# schedule.every(1).minutes.do(job2)

# Keep the script running and execute the jobs
while True:
    schedule.run_pending()
    time.sleep(1)
