import os
import sys
import pandas as pd

report = pd.DataFrame(columns=['obsid', 'job_status', 'jobid', 'status', 'attempts', 'calibration', 'image', 'postImage_0000', 'postImage_0001', 'postImage_0002', 'postImage_0003', 'flagged'])
report.set_index('obsid', inplace=True)
report.to_csv('report.csv')
