import os
import time
import tqdm
import requests
import argparse


URL = "https://www.e-zpassny.com/vector/jcaptcha.do"


if __name__ == '__main__':
    ap = argparse.ArgumentParser()

    ap.add_argument(
        '-o',
        '--output',
        required=True,
        help='path to output directory of images'
    )

    ap.add_argument(
        '-n',
        '--num-images',
        type=int,
        default=500,
        help='number of images to download'
    )

    args = vars(ap.parse_args())

    if not os.path.exists(args['output']):
        os.mkdir(args['output'])

    with tqdm.tqdm(total=args['num_images']) as pbar:
        for i in range(args['num_images']):
            try:
                req = requests.get(URL, timeout=60)

                filename = os.path.sep.join([
                    args['output'],
                    '{}.jpg'.format(
                        str(i).zfill(
                            len(str(args['num_images']))
                        )
                    )
                ])

                with open(filename, 'wb') as fh:
                    fh.write(req.content)

                pbar.update(1)
                pbar.set_description('Success')
            except KeyboardInterrupt as keyboard_interrupt:
                break
            except BaseException as e:
                pbar.update(1)
                pbar.write('Error "%s" downloading image #%d' % (e, i + 1))

            time.sleep(0.1)