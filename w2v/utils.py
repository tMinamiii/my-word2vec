import csv
import ftplib
from w2v import settings


def find_and_load_token_files():
    ftp = ftplib.FTP()
    ftp.encoding = 'utf-8'
    ftp.connect(settings.FTP_SERVER, 21)
    ftp.login(settings.FTP_USER, settings.FTP_PASS)
    for cat in settings.CATEGORIES:
        category_path = '{0}/{1}'.format(settings.FTP_TOKEN_DIR, cat)
        find_result = set()

        def find_ftp(line):
            filename = line.split(' ')[-1]
            filepath = '{0}/{1}'.format(category_path, filename)
            find_result.add(filepath)

        lscmd = 'LIST {}'.format(category_path)
        ftp.retrlines(lscmd, find_ftp)

        for path in find_result:
            retrcmd = 'RETR {}'.format(path)
            byte_list = bytearray()
            ftp.retrbinary(retrcmd, byte_list.extend)
            lines = byte_list.decode('utf-8').split('\n')
            for line in csv.reader(lines):
                if len(line) == 1:
                    yield line[0]
    ftp.quit()
