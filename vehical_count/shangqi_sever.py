import os
import time
import email.parser, email.policy
from shangqi_utils.shangqi_detect import Detect
from shangqi_utils.email_utils import email_initial, get_img, send_email


conn = email_initial()
init_num, _ = conn.stat()
# init_num -= 10

img_save_dir = '/mnt/ActionRecog/dataset/car_count/shangqi_test_rt'
results_dir = '/mnt/ActionRecog/dataset/car_count/shangqi_test_rt_results'
detector = Detect(img_save_dir, results_dir)

while True:

    conn = email_initial()
    curr_num, _ = conn.stat()
    #print('init_email_num: ', init_num, ' ', 'curr_email_num: ', curr_num)

    if 5 <= time.localtime()[3] <= 20:
        for email_id in range(init_num+1, curr_num+1):
            resp, maildata, r = conn.retr(email_id)
            data = b'\r\n'.join(maildata)
            msg = email.parser.BytesParser(policy=email.policy.default).parsebytes(data)
            subject = msg['subject']
            title = 'IKE02' if 'BAOSHA02' in subject else 'IKE01'
            user = str(msg['from'].addresses[0]).split(' ')[1][1:-1]

            if user in ['56971609@qq.com']:
                event_time = get_img(msg, title, img_save_dir)

            if len(os.listdir(img_save_dir)) == 2:
                detector.inference()
                send_email(event_time, results_dir)
                detector.clear()
                print()

    init_num = curr_num
    time.sleep(600)