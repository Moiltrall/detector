from jetson_utils import videoSource, videoOutput, cudaFont, Log
import sys
import argparse


# create video sources & outputs
input = videoSource("", argv=sys.argv)
output = videoOutput("", argv=sys.argv)
font = cudaFont()

while True:
    # capture the next image
    img = input.Capture()

    if img is None:  # timeout
        continue

    #     # classify the image and get the topK predictions
    # # if you only want the top class, you can simply run:
    # #   class_id, confidence = net.Classify(img)
    # predictions = net.Classify(img, topK=args.topK)
    #
    # # draw predicted class labels
    # for n, (classID, confidence) in enumerate(predictions):
    #     classLabel = net.GetClassLabel(classID)
    #     confidence *= 100.0
    #
    #     print(f"imagenet:  {confidence:05.2f}% class #{classID} ({classLabel})")
    #
    #     font.OverlayText(img, text=f"{confidence:05.2f}% {classLabel}",
    #                      x=5, y=5 + n * (font.GetSize() + 5),
    #                      color=font.White, background=font.Gray40)

    # render the image
    output.Render(img)

    # # update the title bar
    # output.SetStatus("{:s} | Network {:.0f} FPS".format(net.GetNetworkName(), net.GetNetworkFPS()))
    #
    # # print out performance info
    # net.PrintProfilerTimes()

    # exit on input/output EOS
    if not input.IsStreaming() or not output.IsStreaming():
        break