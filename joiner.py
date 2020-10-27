import os
import argparse

parser = argparse.ArgumentParser('Joining multiple files in a directory')
parser.add_argument('--file', dest='dest_file', default='join.txt', help='Destination filename')
parser.add_argument('--dir', dest='target_dir', default='files/', help='Target directory')
parser.add_argument('--clear', default = 0, help='Clear files in target directory')
arg = parser.parse_args()


for f in os.listdir(arg.target_dir):
    filename = arg.target_dir + f
    os.system("cat "+filename+" >> "+arg.dest_file)



os.system("perl -pe \'chomp if eof\' "+arg.dest_file)


if arg.clear:
    os.system("rm "+arg.target_dir+"*")
