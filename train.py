from absl import app, flags, logging
from absl.flags import FLAGS
import tensorflow as tf
import numpy as np
import os, shutil
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard
)
from yolov3_tf2.models import (
    YoloV3, YoloV3Tiny, YoloLoss,
    yolo_anchors, yolo_anchor_masks,
    yolo_tiny_anchors, yolo_tiny_anchor_masks
)
from yolov3_tf2.utils import freeze_all
import yolov3_tf2.dataset as dataset

flags.DEFINE_string('dataset', '', 'path to dataset')
flags.DEFINE_boolean('tiny', False, 'yolov3 or yolov3-tiny')
flags.DEFINE_string('weights', './checkpoints/yolov3.tf',
                    'path to weights file')
flags.DEFINE_string('classes', './data/coco.names', 'path to classes file')
flags.DEFINE_string('name', '', 'output file name to save')
flags.DEFINE_string('gpu', '', 'name of gpu to use')
flags.DEFINE_enum('mode', 'fit', ['fit', 'eager_fit', 'eager_tf'],
                  'fit: model.fit, '
                  'eager_fit: model.fit(run_eagerly=True), '
                  'eager_tf: custom GradientTape')
flags.DEFINE_enum('transfer', 'none',
                  ['none', 'darknet', 'no_output', 'frozen', 'fine_tune'],
                  'none: Training from scratch, '
                  'darknet: Transfer darknet, '
                  'no_output: Transfer all but output, '
                  'frozen: Transfer and freeze all, '
                  'fine_tune: Transfer all and freeze darknet only')
flags.DEFINE_integer('size', 416, 'image size')
flags.DEFINE_integer('epochs', 2, 'number of epochs')
flags.DEFINE_integer('batch_size', 8, 'batch size')
flags.DEFINE_float('learning_rate', 1e-3, 'learning rate')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')


def get_free_gpu():
    """Selects the gpu with the most free memory
    """
    import subprocess
    import numpy as np

    output = subprocess.Popen('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free', stdout=subprocess.PIPE,
                              shell=True).communicate()[0]
    output = output.decode("ascii")
    # assumes that it is on the popiah server and the last gpu is not used
    memory_available = [int(x.split()[2]) for x in output.split("\n")[:-2]]
    if not memory_available:
        return
    print("Setting GPU to use to PID {}".format(np.argmax(memory_available)))
    return np.argmax(memory_available)


def set_one_gpu():

    gpu = FLAGS.gpu
    if not gpu:
        gpu = str(get_free_gpu())

    if not gpu:
        return

    print("Using GPU: %s" % gpu)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu


def main(_argv):
    set_one_gpu()

    if FLAGS.tiny:
        model = YoloV3Tiny(FLAGS.size, training=True,
                           classes=FLAGS.num_classes)
        anchors = yolo_tiny_anchors
        anchor_masks = yolo_tiny_anchor_masks
    else:
        model = YoloV3(FLAGS.size, training=True, classes=FLAGS.num_classes)
        anchors = yolo_anchors
        anchor_masks = yolo_anchor_masks

    # train_dataset = dataset.load_fake_dataset()
    dataset_name = 'data/' + FLAGS.dataset + '.train.record'
    val_dataset_name = 'data/' + FLAGS.dataset + '.val.record'

    train_dataset = dataset.load_tfrecord_dataset(
            dataset_name, FLAGS.classes)
    train_dataset = train_dataset.shuffle(buffer_size=1024)  # TODO: not 1024
    train_dataset = train_dataset.batch(FLAGS.batch_size)
    train_dataset = train_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))
    train_dataset = train_dataset.prefetch(
        buffer_size=tf.data.experimental.AUTOTUNE)

    tf_name = FLAGS.name
    if not tf_name:
        tf_name = 'train' + FLAGS.gpu
    best_tf_name = "checkpoints/%s_best.tf" % tf_name
    last_tf_name = "checkpoints/%s_last.tf" % tf_name

    # val_dataset = dataset.load_fake_dataset()
    val_dataset = dataset.load_tfrecord_dataset(
        val_dataset_name, FLAGS.classes)
    val_dataset = val_dataset.batch(FLAGS.batch_size)
    val_dataset = val_dataset.map(lambda x, y: (
        dataset.transform_images(x, FLAGS.size),
        dataset.transform_targets(y, anchors, anchor_masks, 80)))

    if FLAGS.transfer != 'none':
        model.load_weights(FLAGS.weights)
        if FLAGS.transfer == 'fine_tune':
            # freeze darknet
            darknet = model.get_layer('yolo_darknet')
            freeze_all(darknet)
        elif FLAGS.transfer == 'frozen':
            # freeze everything
            freeze_all(model)
        else:
            # reset top layers
            if FLAGS.tiny:  # get initial weights
                init_model = YoloV3Tiny(
                    FLAGS.size, training=True, classes=FLAGS.num_classes)
            else:
                init_model = YoloV3(
                    FLAGS.size, training=True, classes=FLAGS.num_classes)

            if FLAGS.transfer == 'darknet':
                for l in model.layers:
                    if l.name != 'yolo_darknet' and l.name.startswith('yolo_'):
                        l.set_weights(init_model.get_layer(
                            l.name).get_weights())
                    else:
                        freeze_all(l)
            elif FLAGS.transfer == 'no_output':
                for l in model.layers:
                    if l.name.startswith('yolo_output'):
                        l.set_weights(init_model.get_layer(
                            l.name).get_weights())
                    else:
                        freeze_all(l)

    optimizer = tf.keras.optimizers.Adam(lr=FLAGS.learning_rate)
    loss = [YoloLoss(anchors[mask], classes=FLAGS.num_classes)
            for mask in anchor_masks]
    best_val_loss = 0

    if FLAGS.mode == 'eager_tf':
        # Eager mode is great for debugging
        # Non eager graph mode is recommended for real training
        avg_loss = tf.keras.metrics.Mean('loss', dtype=tf.float32)
        avg_val_loss = tf.keras.metrics.Mean('val_loss', dtype=tf.float32)

        for epoch in range(1, FLAGS.epochs + 1):
            for batch, (images, labels) in enumerate(train_dataset):
                with tf.GradientTape() as tape:
                    outputs = model(images, training=True)
                    regularization_loss = tf.reduce_sum(model.losses)
                    pred_loss = []
                    for output, label, loss_fn in zip(outputs, labels, loss):
                        pred_loss.append(loss_fn(label, output))
                    total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                grads = tape.gradient(total_loss, model.trainable_variables)
                optimizer.apply_gradients(
                    zip(grads, model.trainable_variables))

                # logging.info("{}_train_{}, {}, {}".format(
                #     epoch, batch, total_loss.numpy(),
                #     list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_loss.update_state(total_loss)

            for batch, (images, labels) in enumerate(val_dataset):
                outputs = model(images)
                regularization_loss = tf.reduce_sum(model.losses)
                pred_loss = []
                for output, label, loss_fn in zip(outputs, labels, loss):
                    pred_loss.append(loss_fn(label, output))
                total_loss = tf.reduce_sum(pred_loss) + regularization_loss

                # logging.info("{}_val_{}, {}, {}".format(
                #     epoch, batch, total_loss.numpy(),
                #     list(map(lambda x: np.sum(x.numpy()), pred_loss))))
                avg_val_loss.update_state(total_loss)

            val_lost = avg_val_loss.result().numpy()
            logging.info("{}, train: {}, val: {}".format(
                epoch,
                avg_loss.result().numpy(),
                val_lost))

            avg_loss.reset_states()
            avg_val_loss.reset_states()
            model.save_weights(last_tf_name)
            if best_val_loss == 0 or best_val_loss > val_lost:
                best_val_loss = val_lost
                logging.info("saving best val loss.")
                model.save_weights(best_tf_name)
    else:
        model.compile(optimizer=optimizer, loss=loss,
                      run_eagerly=(FLAGS.mode == 'eager_fit'))

        callbacks = [
            ReduceLROnPlateau(verbose=1),
            EarlyStopping(patience=3, verbose=1),
            ModelCheckpoint(best_tf_name,
                            verbose=1, save_weights_only=True),
            TensorBoard(log_dir='logs')
        ]

        history = model.fit(train_dataset,
                            epochs=FLAGS.epochs,
                            callbacks=callbacks,
                            validation_data=val_dataset)

    if history is not None:
        print(history.history['val_loss'])
        best_val_loss = min(history.history['val_loss'])
        model.save_weights(best_tf_name)

    print("Best weights are saved as %s" % best_tf_name)
    tiny = 'tiny_' if FLAGS.tiny else ''
    out_name = "%s_d%s_%sm%s_bs%d_s%s_e%d_val%d" % \
         (tf_name, FLAGS.dataset, tiny, FLAGS.transfer, FLAGS.batch_size, FLAGS.size, FLAGS.epochs, best_val_loss)
    mfn = "data/model/%s.h5" % out_name

    final_tf_name = "%s.tf" % out_name
    copy_tf("%s_best.tf" % tf_name, final_tf_name)
    print("Final checkpoint file saved as: %s" % final_tf_name)
    model.save(mfn, save_format='tf')
    print("Model file saved to: %s" % mfn)


def copy_tf(ifn, ofn):
    for fn in os.listdir('checkpoints'):
        if not fn.startswith(ifn):
            continue
        out = fn.replace(ifn, ofn)
        shutil.copyfile('checkpoints/' + fn, 'checkpoints/' + out)

if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass
