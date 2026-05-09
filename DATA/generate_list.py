import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--name', default='PST', type=str)
parser.add_argument('--path', default=r'D:\ILSVRC2012_img_train', type=str)
args = parser.parse_args()

folders = os.listdir(args.path)

if not os.path.exists("list_IMAGENET"):
    os.makedirs("list_IMAGENET")

fl = open('list_IMAGENET/' + args.name + '_list.txt', 'w')
fn = open('list_IMAGENET/' + args.name + '_name.txt', 'w')

for i, folder in enumerate(folders):

    fn.write(str(i) + ' ' + folder + '\n')

    folder_path = os.path.join(args.path, folder)
    files = os.listdir(folder_path)

    for file in files:

        fl.write('{} {}\n'.format(os.path.join(folder_path, file), i))

fl.close()
fn.close()
