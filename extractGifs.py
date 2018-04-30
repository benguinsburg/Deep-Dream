import os
from PIL import Image


def extractFrames(inGif, outFolder):
    frame = Image.open(inGif)
    nframes = 0
    mypalette = frame.getpalette()


    while frame:
        frame.putpalette(mypalette)

        # frame.save( '%s/%s-%s.gif' % (outFolder, os.path.basename(inGif), nframes ) , 'JPEG')


        new_im = Image.new("RGB", frame.size)
        new_im.paste(frame)
        new_im.save('%s/%s-%s.jpg' % (outFolder, nframes, os.path.basename(inGif) ) , 'JPEG')

        nframes += 1
        try:
            frame.seek( nframes )
        except EOFError:
            break;
    return True
    

