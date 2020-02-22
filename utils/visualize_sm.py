import cPickle
from PIL import Image
import argparse

if __name__ == "__main__":
      parser = argparse.ArgumentParser(description='Visualize sm file')
      parser.add_argument('image')
      args = parser.parse_args()

      filename = 'GTsegmask_VOC_2012_train/{}'.format(args.image)
      with open(filename, 'rb') as f:
            seg_mask = cPickle.load(f)

      seg_mask = (seg_mask*255).astype('uint8')
      img = Image.fromarray(seg_mask).convert('RGB')
      print "image size: ", img.size
      img.show()

