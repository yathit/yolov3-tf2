import time
from absl import app, flags, logging
from absl.flags import FLAGS
import cv2, os, math
import numpy as np
import tensorflow as tf
from yolov3_tf2.models import (
  YoloV3, YoloV3Tiny
)
from yolov3_tf2.dataset import transform_images
from yolov3_tf2.utils import draw_outputs

flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('size', 416, 'resize images to')
flags.DEFINE_boolean('save', False, 'save output file')
flags.DEFINE_boolean('all', False, 'run all data')
flags.DEFINE_string('image', './data/girl.png', 'path to input image')
flags.DEFINE_string('output', './output.jpg', 'path to output image')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def main(_argv):
  physical_devices = tf.config.experimental.list_physical_devices('GPU')
  if len(physical_devices) > 0:
    tf.config.experimental.set_memory_growth(physical_devices[0], True)

  if FLAGS.tiny:
    yolo = YoloV3Tiny(classes=FLAGS.num_classes)
  else:
    yolo = YoloV3(classes=FLAGS.num_classes)

  yolo.load_weights(FLAGS.weights)
  logging.info('weights loaded')

  class_names = [c.strip() for c in open(FLAGS.classes).readlines()]
  logging.info('classes loaded')

  nfp = 0
  npp = 0
  nxp = 0
  num_prediction = 0
  num_correct_predction = 0

  for raw in tf.data.TFRecordDataset(['data/test.record', 'data/train.record', 'data/val.record']):
    record = tf.train.Example()
    record.ParseFromString(raw.numpy())
    name = record.features.feature['image/filename'].bytes_list.value[0].decode("utf-8")
    fn = name[name.index('/') + 1:]
    print("######################### Record", name, "#########################")

    lc = fn[len(fn) - 5]
    test_data = lc == '0'

    if not os.path.exists('data/' + name):
      continue

    img = tf.image.decode_image(open('data/' + name, 'rb').read(), channels=3)
    img = tf.expand_dims(img, 0)
    img = transform_images(img, FLAGS.size)
    wh = np.flip(img.shape[0:2])

    xmin = record.features.feature['image/object/bbox/xmin'].float_list.value
    xmax = record.features.feature['image/object/bbox/xmax'].float_list.value
    ymin = record.features.feature['image/object/bbox/ymin'].float_list.value
    ymax = record.features.feature['image/object/bbox/ymax'].float_list.value
    xx = []
    yy = []
    for i in range(0, len(xmin)):
      xx.append(wh[0] * (xmin[i] + xmax[i]) / 2)
      yy.append(wh[1] * (ymin[i] + ymax[i]) / 2)


    boxes, scores, classes, nums = yolo(img)

    num_prediction += len(xmin)
    threshold = 20


    nfp += 1
    if nums[0] > 0:
      npp += 1
      img = cv2.imread('data/scaled/' + fn)
      img = draw_outputs(img, (boxes, scores, classes, nums), class_names)
      ofn = 'data/predict/' + fn
      if FLAGS.save or test_data:
        cv2.imwrite(ofn, img)
        logging.info('output saved to: {}'.format(ofn))
      for i in range(0, len(xx)):
        for j in range(nums[0]):
          if classes[0][j] != i:
            continue
          x1y1 = (np.array(boxes[0][j][0:2]) * wh).astype(np.int32)
          x2y2 = (np.array(boxes[0][j][2:4]) * wh).astype(np.int32)

          x = (x1y1[0] + x2y2[0]) / 2
          y = (x1y1[1] + x2y2[1]) / 2
          d = math.sqrt(math.pow((x - xx[i]), 2) + math.pow((y - yy[i]), 2))
          if d < threshold:
            num_correct_predction += 1
            break


    logging.info('detections')
    for i in range(nums[0]):
      logging.info('\t{}, {}, {}'.format(class_names[int(classes[0][i])],
                                         np.array(scores[0][i]),
                                         np.array(boxes[0][i])))

    if nums[0] == 4:
      nxp += 1

  print("%d processed. %d some prediction. %d (%1.0f %%) has complete prediction" % (nfp, npp, nxp, 100 * nxp / nfp))

  print("%1.0f %% correctly predicted" % (100 * num_correct_predction / num_prediction))


if __name__ == '__main__':
  try:
    app.run(main)
  except SystemExit:
    pass
