import os
import re
import poplib
import smtplib

from email.header import Header
from email.utils import parseaddr, formataddr
from email.mime.text import MIMEText
from email.mime.base import MIMEBase
from email.mime.multipart import MIMEMultipart
from email import encoders

from_addr = '1129527544@qq.com'
password = 'qenyqjwhdbsfihgb'
to_addr = ['zhengbo.wei@ike-data.com', 'wenjie.ren@ike-data.com']
smtp_server = "smtp.qq.com"
cam_dict = {'IKE01': 'Camera_1.jpg', 'IKE02': 'Camera_2.jpg'}

def email_initial():
    conn = poplib.POP3_SSL('pop.qq.com', 995)
    conn.user(from_addr)
    conn.pass_(password)
    return conn


def get_img(msg, mode=None, img_save_dir=None):
    for part in msg.walk():
        if part.get_content_type().split("/", 1)[0] == 'multipart':
            continue
        elif part.get_content_type().split("/", 1)[0] == 'text':
            body = part.get_content()
            index = re.search('EVENT TIME: ', body, flags=0).span()
            event_time = body[index[0]: index[1]+19]
            print(event_time)
        else:
            filename = cam_dict[mode]
            fp = os.path.join(img_save_dir, filename)
            with open(fp, 'wb') as f:
                f.write(part.get_payload(decode=True))
    return event_time

def _format_addr(s):
    name, addr = parseaddr(s)
    return formataddr((Header(name, "utf-8").encode(), addr))

def send_email(event_time, save_dir):
    msg = MIMEMultipart()
    msg["From"] = _format_addr("ucloud <%s>" % from_addr)
    msg["To"] = _format_addr("ike <%s>" % to_addr)
    msg["Subject"] = Header("Number of Cars", "utf-8").encode()
    msg.attach(MIMEText(event_time, "plain", "utf-8"))

    for fn in os.listdir(save_dir):
        fp = os.path.join(save_dir, fn)
        suffix = fn.split('.')[-1]
        with open(fp, 'rb') as f:
            if suffix == 'jpg':
                mime = MIMEBase("image", "jpg", filename=fn)
            elif suffix == 'png':
                mime = MIMEBase("image", "png", filename=fn)
            elif suffix == 'xlsx':
                mime = MIMEBase("file", "xlsx", filename=fn)
            mime.add_header("Content-Disposition", "attchment", filename=fn)
            mime.add_header("Content-ID", "<0>")
            mime.add_header("X-Attachment-Id", "0")
            mime.set_payload(f.read())
            encoders.encode_base64(mime)
            msg.attach(mime)
    server = smtplib.SMTP_SSL(smtp_server, 465)
    server.set_debuglevel(0)
    server.login(from_addr, password)
    server.sendmail(from_addr, to_addr, msg.as_string())
    server.quit()
    print('Send email success!')