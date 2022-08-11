
# single scale and single crop, it only forwards once per image.  The short edge of the image will be resized to 1080.
python tools/test.py --config test_720_ss

# single scale and single crop, it only forwards once per image. The short edge of the image will be resized to 1080.
#python tools/test.py --config confgs/test_1080_ss


# single scale and mutiple crops. Each image will be splits to 4 crops, which are fed to the model for multiple forwards. The performance and details 
# will be much better than previous methods. The short edge of the image will be resized to 720.
#python tools/test.py --config confgs/test_720_sm

# single scale and mutiple crops. Each image will be splits to 4 crops, which are fed to the model for multiple forwards. The performance and details 
# will be much better than previous methods. The short edge of the image will be resized to 720.
#python tools/test.py --config confgs/test_1080_sm
