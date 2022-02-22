import os
import shutil
import time
import sys


class Recorder(object):
    def __init__(self, snapshot_pref, exclude_dirs=None, max_file_size=10):
        """
        :param snapshot_pref: The dir you want to save the backups
        :param exclude_dirs: The dir name you want to exclude; eg ["results", "data"]
        :param max_file_size: The minimum size of backups file; unit is MB
        """
        date = time.strftime('%Y%m%d%H%M%S', time.localtime(time.time()))
        if not os.path.isdir(snapshot_pref):
            os.makedirs(snapshot_pref, exist_ok=True)
        self.save_path = snapshot_pref
        self.log_file = self.save_path + "log.txt"
        self.code_path = os.path.join(self.save_path, "code_{}/".format(date))
        self.exclude_dirs = exclude_dirs
        self.max_file_size = max_file_size
        os.makedirs(self.code_path, exist_ok=True)
        self.copy_code(dst=self.code_path)

    def copy_code(self, src="./", dst="./code/"):
        start_time = time.time()
        file_abs_list = []
        src_abs = os.path.abspath(src)
        for root, dirs, files in os.walk(src_abs):
            exclude_flag = True in [root.find(exclude_dir)>=0 for exclude_dir in self.exclude_dirs]
            if not exclude_flag:
                for name in files:
                    file_abs_list.append(root + "/" + name)

        for file_abs in file_abs_list:
            file_split = file_abs.split("/")[-1].split('.')
            # if len(file_split) >= 2 and file_split[1] == "py":
            if os.path.getsize(file_abs) / 1024 / 1024 < self.max_file_size and not file_split[-1] == "pyc":
                src_file = file_abs
                dst_file = dst + file_abs.replace(src_abs, "")
                os.makedirs(os.path.dirname(dst_file), exist_ok=True)
                # shutil.copyfile(src=src_file, dst=dst_file)
                try:
                    shutil.copy2(src=src_file, dst=dst_file)
                except:
                    print("copy file error")
        print("|===>Backups using time: %.3f s"%(time.time() - start_time))
    
    def tee_stdout(self, log_path):
        log_file = open(log_path, 'a', 1)
        stdout = sys.stdout

        class Tee:

            def write(self, string):
                log_file.write(string)
                stdout.write(string)

            def flush(self):
                log_file.flush()
                stdout.flush()

        sys.stdout = Tee()