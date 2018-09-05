# DeepDream
# Deep-Dream

usage: DeepDream.py [-h] [-name NAME] [-params PARAMS [PARAMS ...]]
                    [-rescale RESCALE] [-mode MODE] [-c] [-l] [-r]

Run deep dream algorithm on an image.


optional arguments:
  -h, --help            show this help message and exit
  -name NAME            name of the image in im_lib
  -params PARAMS [PARAMS ...]
                        [layer] [subset start] [subset end
                        
  -rescale RESCALE      scale factor for input image
  
  -mode MODE            image, gif, image_as_mp4
  
  -c                    get contrasted image
  
  -l                    load previous random tensor set
  
  -r                    use random distribution of tensors
