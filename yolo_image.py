import sys
import argparse
from yolo import YOLO, detect_image

def _main(args):
    if args.input:
        detect_image(YOLO(**vars(args)), args.close_session, args.input, args.output)
    else:
        print('Must specify at least image_input_path. See usage with --help.')

if __name__ == '__main__':
    # class YOLO defines the default value, so suppress any default here
    parser = argparse.ArgumentParser(argument_default=argparse.SUPPRESS)
    '''
    Command line options
    '''
    parser.add_argument(
        '--model_path', type=str,
        aliases=['model'],
        help='path to model weight file, default ' + YOLO.get_defaults('model_path')
    )

    parser.add_argument(
        '--anchors_path', type=str,
        aliases=['anchors'],
        help='path to anchor definitions, default ' + YOLO.get_defaults('anchors_path')
    )

    parser.add_argument(
        '--classes_path', type=str,
        aliases=['classes'],
        help='path to class definitions, default ' + YOLO.get_defaults('classes_path')
    )

    parser.add_argument(
        '--font_path', type=str,
        aliases=['font'],
        help='path to font used to write labels in images, default ' + YOLO.get_defaults('font_path')
    )

    parser.add_argument(
        '--gpu_num', type=int,
        default=1,
        help='Number of GPU to use, default ' + str(YOLO.get_defaults('gpu_num'))
    )

    parser.add_argument(
        '--close_session', type=bool,
        help='Indicates if session must be close after prediction. Useful when it is run in colab to avoid session close',
        action='store_true'
    )

    '''
    Command line positional arguments
    '''
    parser.add_argument(
        '--input', type=str,
        required=True,
        help = 'Image input path'
    )

    parser.add_argument(
        '--output', type=str,
        required=True,
        help = 'Image output path'
    )

    _main(parser.parse_args())
