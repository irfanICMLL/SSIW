
# single scale and single crop, it only forwards once per image.  The short edge of the image will be resized to 1080.
python tools/test.py --config test_720_ss

# use the label user identified for semantic segmentation
#python tools/test.py --config test_720_ss --user_label dog mouse horse rug_floormat wall person vegetation pizza

# add new label definitions identified by user
#python tools/test.py --config test_720_ss --new_definitions="{'duck': 'This is an image of duck bird. Ducks a group of warm-blooded vertebrates constituting the class Aves, with a flat bill and yellow webbed feet.', 'deer': 'This is an image of deer, similar to sheep or dog.'}"

# single scale and single crop, it only forwards once per image. The short edge of the image will be resized to 1080.
#python tools/test.py --config confgs/test_1080_ss

# single scale and mutiple crops. Each image will be splits to 4 crops, which are fed to the models for multiple forwards. The performance and details
# will be much better than previous methods. The short edge of the image will be resized to 720.
#python tools/test.py --config confgs/test_720_sm

# single scale and mutiple crops. Each image will be splits to 4 crops, which are fed to the models for multiple forwards. The performance and details
# will be much better than previous methods. The short edge of the image will be resized to 720.
#python tools/test.py --config confgs/test_1080_sm