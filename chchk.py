from absl import app, flags
import shutil, os

flags.DEFINE_string('i', ' yolov3_train_25', 'input checkpoint name')
flags.DEFINE_string('o', ' yolov3_train_25', 'input checkpoint name')


def main(_argv):

    ifn = flags.FLAGS.i.strip()
    ofn = flags.FLAGS.o.strip()
    for fn in os.listdir('checkpoints'):
        if not fn.startswith(ifn):
            continue
        out = fn.replace(ifn, ofn)
        print(fn + ' --> ' + out)
        shutil.copyfile('checkpoints/' + fn, 'checkpoints/' + out)


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
