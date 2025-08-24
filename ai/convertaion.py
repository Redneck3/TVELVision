import tensorflow as tf
import os
import glob
import xml.etree.ElementTree as ET
from object_detection.utils import dataset_util

# Словарь классов (можете добавить ещё, если будут другие дефекты)
label_map = {"tablet": 1}

def create_tf_example(xml_file, image_dir):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    filename = root.find('filename').text
    img_path = os.path.join(image_dir, filename)
    with tf.io.gfile.GFile(img_path, 'rb') as fid:
        encoded_jpg = fid.read()

    size = root.find('size')
    width = int(size.find('width').text)
    height = int(size.find('height').text)

    xmins = []
    xmaxs = []
    ymins = []
    ymaxs = []
    classes_text = []
    classes = []

    for member in root.findall('object'):
        class_name = member.find('name').text
        bndbox = member.find('bndbox')
        xmin = int(bndbox.find('xmin').text) / width
        ymin = int(bndbox.find('ymin').text) / height
        xmax = int(bndbox.find('xmax').text) / width
        ymax = int(bndbox.find('ymax').text) / height

        xmins.append(xmin)
        ymins.append(ymin)
        xmaxs.append(xmax)
        ymaxs.append(ymax)
        classes_text.append(class_name.encode('utf8'))
        classes.append(label_map[class_name])

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/source_id': dataset_util.bytes_feature(filename.encode('utf8')),
        'image/encoded': dataset_util.bytes_feature(encoded_jpg),
        'image/format': dataset_util.bytes_feature(b'jpg'),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def generate_tfrecord(annotations_dir, image_dir, output_path):
    writer = tf.io.TFRecordWriter(output_path)
    for xml_file in glob.glob(os.path.join(annotations_dir, '*.xml')):
        tf_example = create_tf_example(xml_file, image_dir)
        writer.write(tf_example.SerializeToString())
    writer.close()
    print(f"TFRecord saved to {output_path}")


if __name__ == '__main__':
    annotations_dir = r"E:/Code/TVELVision/ai/dataset/tablet"  # папка с XML
    image_dir = r"E:/Code/TVELVision/ai/dataset/tablet"        # папка с JPG
    output_path = r"E:/Code/TVELVision/ai/tablet.tfrecord"
    generate_tfrecord(annotations_dir, image_dir, output_path)
